"""
train_pt.py  —  H-Net · treinamento em português (ou qualquer corpus HF)

Suporta dois tipos de núcleo (rede central), escolhidos automaticamente pelo
`arch_layout` do --model-config:

  1) Núcleo nativo H-Net (ex.: "T22", "T26"):
     ["m4", ["T22"], "m4"]
     ["m4", ["T1m4", ["T26"], "m4T1"], "m4"]
     -> treina a hierarquia inteira normalmente (comportamento original).

  2) Núcleo Llama 3.2 3B (spec "Llama"):
     ["m4", ["Llama"], "m4"]
     ["m4", ["T1m4", ["Llama"], "m4T1"], "m4"]
     -> carrega os pesos de um Llama 3.2 3B pré-treinado (HF) dentro do
        submódulo indicado por --llama-attr-path, congela esse submódulo
        (requires_grad=False) e treina só as redes externas (m4 / T1m4,
        i.e. o "tokenizador" do H-Net).
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from hnet.utils.train import load_balancing_loss, group_params
from hnet.utils.csv_logger import CsvLogger
from hnet.utils.arch import count_boundary_stages
from hnet.utils.metrics import (
    HNetMetrics,
    compute_bpic,
    compute_compression_ratio,
    compute_ratio_loss,
)
from hnet.models.mixer_seq import build_model, find_frozen_external_modules
from train.evaluate import evaluate
from hnet.utils.datasets import ByteConcatDataset, collate_fn
from hnet.utils.distributed import setup_distributed
load_dotenv()
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--dataset-name",        type=str, default="uonlp/CulturaX")
    parser.add_argument("--dataset-config-name", type=str, default="pt")
    parser.add_argument("--dataset-split",       type=str, default="train")
    parser.add_argument("--val-split",           type=str, default=None)
    parser.add_argument("--text-column",         type=str, default="text")
    parser.add_argument("--streaming",           dest="streaming", action="store_true",  default=True)
    parser.add_argument("--no-streaming",        dest="streaming", action="store_false")
    parser.add_argument("--trust-remote-code",   action="store_true", default=False)

    # modelo
    parser.add_argument("--model-config",  type=str, required=True)
    parser.add_argument("--resume-from",   type=str, default=None)

    # treino
    parser.add_argument("--seq-len",              type=int,   default=4096)
    parser.add_argument("--batch-size",           type=int,   default=8)
    parser.add_argument("--grad-accum-steps",     type=int,   default=1)
    parser.add_argument("--max-steps",            type=int,   default=None)
    parser.add_argument("--max-tokens",           type=int,   default=None)
    parser.add_argument("--warmup-steps",         type=int,   default=1000)
    parser.add_argument("--lr",                   type=float, default=3e-4)
    parser.add_argument("--min-lr",               type=float, default=3e-5)
    parser.add_argument("--weight-decay",         type=float, default=0.1)
    parser.add_argument("--grad-clip",            type=float, default=1.0)
    parser.add_argument("--lr-multiplier",        type=str,   default=None)
    parser.add_argument("--load-balancing-n",     type=str,   default=None)
    parser.add_argument("--load-balancing-weight",type=float, default=0.03)

    # infra
    parser.add_argument("--num-workers",               type=int, default=2)
    parser.add_argument("--dtype",                     type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--log-every",                 type=int, default=10)
    parser.add_argument("--save-every",                type=int, default=1000)
    parser.add_argument("--eval-every",                type=int, default=500)
    parser.add_argument("--eval-steps",                type=int, default=50)
    parser.add_argument("--out-dir",                   type=str, default="checkpoints/pt-hnet")
    parser.add_argument("--csv-path",                  type=str, default=None)
    parser.add_argument("--seed",                      type=int, default=0)
    parser.add_argument("--ddp-find-unused-parameters",action="store_true", default=False)

    # benchmark
    parser.add_argument("--benchmark", action="store_true", default=False)

    args = parser.parse_args()
    if args.dataset_config_name in (None, "None", ""):
        args.dataset_config_name = None

    if args.max_steps is None and args.max_tokens is None:
        args.max_steps = 100_000  # default se nenhum for passado
    elif args.max_steps is not None and args.max_tokens is not None:
        raise ValueError("Passe apenas --max-steps OU --max-tokens, não ambos")

    use_token_limit = args.max_tokens is not None

    if args.benchmark:
        torch.cuda.reset_peak_memory_stats()

    # distribuído
    rank, world_size, local_rank, is_distributed = setup_distributed()
    is_main = rank == 0

    torch.manual_seed(args.seed + rank)
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
    if is_distributed:
        dist.barrier()

    device = f"cuda:{local_rank}" if is_distributed else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    if is_main:
        print(f"Distribuído: {is_distributed} | world_size={world_size} | device={device}")

    # dataset
    dataset = ByteConcatDataset(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        split=args.dataset_split,
        text_column=args.text_column,
        seq_len=args.seq_len,
        streaming=args.streaming,
        seed=args.seed,
        hf_token=HF_TOKEN,
        trust_remote_code=args.trust_remote_code,
        rank=rank,
        world_size=world_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        collate_fn=collate_fn, num_workers=args.num_workers)

    val_loader = None
    if args.val_split and is_main:
        val_dataset = ByteConcatDataset(
            dataset_name=args.dataset_name,
            dataset_config_name=args.dataset_config_name,
            split=args.val_split,
            text_column=args.text_column,
            seq_len=args.seq_len,
            streaming=args.streaming,
            seed=args.seed,
            hf_token=HF_TOKEN,
            trust_remote_code=args.trust_remote_code,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                collate_fn=collate_fn, num_workers=0)

    # modelo
    if is_main:
        print("Construindo modelo...")
    model, hnet_cfg = build_model(args.model_config, device=device, dtype=dtype)
    n_boundary_stages = count_boundary_stages(hnet_cfg.arch_layout)
    n_total_stages    = n_boundary_stages + 1

    if is_main:
        print(f"Hierarquia: {n_total_stages} estágio(s), {n_boundary_stages} módulo(s) de roteamento")

    start_step = 0
    if args.resume_from:
        if is_main:
            print(f"Retomando de {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        # Na retomada:
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            # missing = pesos do Llama que estão em memória mas não em disco (esperado)
            # unexpected = vazio
            start_step = ckpt.get("step", 0)
            total_tokens = ckpt.get("total_tokens", 0)
        if is_main:
            print(f"Retomando a partir do passo {start_step}")

    if is_main:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"Parâmetros treináveis: {n_trainable/1e6:.1f}M | congelados: {n_frozen/1e6:.1f}M")

    # lr multipliers
    lr_mult = (
        [float(x) for x in args.lr_multiplier.split(",")]
        if args.lr_multiplier
        else [1.0] * n_total_stages
    )
    assert len(lr_mult) == n_total_stages
    model.apply_lr_multiplier(lr_mult)

    # N por estágio de roteamento (compressão alvo)
    lb_n = (
        [float(x) for x in args.load_balancing_n.split(",")]
        if args.load_balancing_n
        else [4.0] * n_boundary_stages
    )
    assert len(lb_n) == n_boundary_stages

    # CSV — criado depois de conhecer n_boundary_stages
    csv_logger = None
    if is_main:
        csv_path   = args.csv_path or str(Path(args.out_dir) / "metrics.csv")
        csv_logger = CsvLogger(csv_path, n_stages=n_boundary_stages)
        print(f"Métricas em {csv_path}")

    # otimizador — parâmetros congelados (núcleo Llama) ficam fora dos grupos
    param_groups = group_params(model)
    for g in param_groups:
        g["params"] = [p for p in g["params"] if p.requires_grad]
        g.setdefault("weight_decay", args.weight_decay)
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    raw_model = model
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=args.ddp_find_unused_parameters)

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * step / max(1, args.warmup_steps)
        # Progress ainda é baseado em steps, não tokens
        # (se quiser variar por tokens, seria mais complexo)
        max_steps_for_schedule = args.max_steps or 100_000
        progress = (step - args.warmup_steps) / max(1, max_steps_for_schedule - args.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return args.min_lr + (args.lr - args.min_lr) * cosine

    # ── acumuladores ────────────────────────────────────────────────────────
    # Detectar módulos congelados que já vêm do config
    frozen_modules = find_frozen_external_modules(model)
    if frozen_modules and is_main:
        print(f"{len(frozen_modules)} backbone(s) externo(s) pré-treinado(s) e congelado(s).")

    model.train()
    for m in frozen_modules:
        m.eval()

    step             = start_step
    total_tokens     = 0
    t0               = time.time()
    train_start      = t0
    steps_since_log  = 0

    train_metrics = HNetMetrics()   # acumula a janela de log (resetado a cada args.log_every)

    optimizer.zero_grad()
    data_iter = iter(loader)

    # ── loop principal ───────────────────────────────────────────────────────
    while True:
        # Condição de parada: max_steps ou max_tokens
        if use_token_limit:
            if total_tokens >= args.max_tokens:
                break
        else:
            if step >= args.max_steps:
                break

        for _ in range(args.grad_accum_steps):
            try:
                input_ids, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                input_ids, targets = next(data_iter)

            input_ids = input_ids.to(device)
            targets   = targets.to(device)
            B, L0     = input_ids.shape

            tokens_this_batch = B * L0
            total_tokens += tokens_this_batch

            output = model(input_ids)
            logits = output.logits

            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1),
            )

            # lb_loss original (fallback para modelos sem boundary_probs_list)
            lb_loss = torch.zeros((), device=device)
            for bpred, n in zip(output.bpred_output, lb_n):
                lb_loss = lb_loss + load_balancing_loss(bpred, n)

            # ratio_loss (H-Net com dynamic chunking)
            ratio_loss   = torch.zeros((), device=device)
            stage_ratios = {}
            bpic         = 0.0
            has_hnet     = (
                hasattr(output, "boundary_probs_list") and
                hasattr(output, "boundary_ind_list")
            )
            if has_hnet and lb_n:
                for b_probs, b_inds, N in zip(
                    output.boundary_probs_list, output.boundary_ind_list, lb_n
                ):
                    ratio_loss = ratio_loss + compute_ratio_loss(b_probs, b_inds, N)
                lb_loss = ratio_loss   # substitui o lb_loss genérico

                for s, b_ind in enumerate(output.boundary_ind_list):
                    key = f"compression_L{s+1}/L{s}"
                    stage_ratios[key] = compute_compression_ratio(b_ind)

                bpic = compute_bpic(L0, output.boundary_ind_list)

            loss = lm_loss + args.load_balancing_weight * lb_loss
            (loss / args.grad_accum_steps).backward()

            # acumula métricas (sem grad) — cada microbatch é um update()
            with torch.no_grad():
                train_metrics.update(
                    nll_sum_nats=lm_loss.item() * B * L0,
                    n_tokens=B * L0,
                    lb_loss=lb_loss.item(),
                    ratio_loss=ratio_loss.item(),
                    bpic=bpic,
                    stage_ratios=stage_ratios,
                )

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        lr = lr_at(step)
        for g in optimizer.param_groups:
            g["lr"] = lr * g.get("lr_multiplier", 1.0)

        optimizer.step()
        optimizer.zero_grad()
        steps_since_log += 1
        step            += 1

        # ── log de treino ────────────────────────────────────────────────────
        if steps_since_log >= args.log_every:
            dt = time.time() - t0

            if is_distributed:
                # operação coletiva: todos os ranks precisam chamar
                train_metrics.all_reduce_(device)

            if is_main:
                avgs = train_metrics.averages()
                if avgs:
                    n            = steps_since_log
                    avg_lm       = avgs["lm_loss"]
                    avg_lb       = avgs["lb_loss"]
                    avg_ratio    = avgs["ratio_loss"]
                    avg_bpb      = avgs["bpb"]
                    avg_bpic     = avgs["bpic"]
                    avg_ratios   = avgs["stage_ratios"]
                    perplexity   = avgs["perplexity"]
                    tok_per_step = args.batch_size * args.seq_len * args.grad_accum_steps * world_size
                    tok_per_sec  = tok_per_step * n / max(dt, 1e-8)

                    ratio_str = " | ".join(f"{k}={v:.3f}" for k, v in avg_ratios.items())
                    print(
                        f"passo {step:6d} | tokens {total_tokens:6d} | lm {avg_lm:.4f} | ppl {perplexity:.2f}"
                        f" | bpb {avg_bpb:.4f} | bpic {avg_bpic:.2f}"
                        f" | ratio_loss {avg_ratio:.4f} | lb {avg_lb:.4f}"
                        + (f" | {ratio_str}" if ratio_str else "")
                        + f" | lr {lr:.2e} | {dt/n:.2f}s/step | {tok_per_sec:,.0f} tok/s"
                    )
                    if is_main and csv_logger:
                        csv_logger.log(
                            step=step,
                            split="train",
                            wall_time=round(time.time() - train_start, 2),
                            lm_loss=avg_lm,
                            perplexity=perplexity,
                            lb_loss=avg_lb,
                            ratio_loss=avg_ratio,
                            total_loss=avg_lm + args.load_balancing_weight * avg_lb,
                            bpb=avg_bpb,
                            bpic=avg_bpic,
                            lr=lr,
                            tokens_per_sec=round(tok_per_sec, 1),
                            tokens=total_tokens,
                            **avg_ratios,
                        )

            # reset (em todos os ranks, para manter a janela sincronizada)
            train_metrics.reset()
            steps_since_log = 0
            t0 = time.time()

        # ── validação ────────────────────────────────────────────────────────
        if is_main and val_loader and step > 0 and step % args.eval_every == 0:
            result = evaluate(raw_model, lambda: iter(val_loader), lb_n, device, args.eval_steps)
            val_lm, val_lb, val_ratio, val_bpb, val_bpic, val_stage_ratios = result

            if val_lm is not None:
                val_ppl = math.exp(min(val_lm, 20))
                ratio_str = " | ".join(f"{k}={v:.3f}" for k, v in val_stage_ratios.items())
                print(
                    f"          [val] passo {step:6d} | tokens {total_tokens:6d}"
                    f" | lm {val_lm:.4f} | ppl {val_ppl:.2f}"
                    f" | bpb {val_bpb:.4f} | bpic {val_bpic:.2f}"
                    f" | ratio_loss {val_ratio:.4f}"
                    + (f" | {ratio_str}" if ratio_str else "")
                )
                if is_main and csv_logger:
                    csv_logger.log(
                        step=step,
                        split="val",
                        wall_time=round(time.time() - train_start, 2),
                        lm_loss=val_lm,
                        perplexity=val_ppl,
                        lb_loss=val_lb,
                        ratio_loss=val_ratio,
                        total_loss=val_lm + args.load_balancing_weight * val_lb,
                        bpb=val_bpb,
                        bpic=val_bpic,
                        tokens=total_tokens,
                        **val_stage_ratios,
                    )

        # ── checkpoint ───────────────────────────────────────────────────────
        if is_main and step > 0 and step % args.save_every == 0:
            ckpt_path  = Path(args.out_dir) / f"step_{step}.pt"
            # No checkpoint:
            torch.save(
                {
                    "model_state_dict": raw_model.state_dict(),
                    "step": step,
                    "total_tokens": total_tokens,
                },
                ckpt_path,
            )
            print(f"Checkpoint salvo em {ckpt_path}")

    if is_main:
        final = Path(args.out_dir) / "final.pt"
        torch.save({
            "model_state_dict": raw_model.state_dict(),
            "step": step,
            "total_tokens": total_tokens,
        }, final)
        print(f"Treino concluído. Checkpoint final em {final}")
        if csv_logger:
            csv_logger.close()

    if args.benchmark:
        if (is_distributed):
            print(
                f"BENCHMARK_RANK={torch.distributed.get_rank()}",
                flush=True,
            )
        torch.cuda.synchronize()

        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

        print(
            f"BENCHMARK_VRAM_ALLOCATED_MB={max_allocated:.0f}",
            flush=True,
        )

        print(
            f"BENCHMARK_VRAM_RESERVED_MB={max_reserved:.0f}",
            flush=True,
        )

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()