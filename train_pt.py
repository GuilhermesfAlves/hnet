"""
train_pt.py  —  H-Net · treinamento em português (ou qualquer corpus HF)
"""

import argparse
import csv
import math
import json
import os
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader
from dotenv import load_dotenv

from datasets import load_dataset

from hnet.models.config_hnet import AttnConfig, SSMConfig, HNetConfig
from hnet.models.mixer_seq import HNetForCausalLM
from hnet.utils.train import load_balancing_loss, group_params
from hnet.utils import ByteTokenizer

load_dotenv()
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")


# --------------------------------------------------------------------------- #
# Distribuído
# --------------------------------------------------------------------------- #
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ByteConcatDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_config_name: Optional[str],
        split: str,
        text_column: str,
        seq_len: int,
        streaming: bool = True,
        shuffle_buffer_size: int = 10_000,
        seed: int = 0,
        hf_token: Optional[str] = None,
        add_doc_boundaries: bool = True,
        trust_remote_code: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.dataset_name        = dataset_name
        self.dataset_config_name = dataset_config_name
        self.split               = split
        self.text_column         = text_column
        self.seq_len             = seq_len
        self.streaming           = streaming
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed                = seed
        self.hf_token            = hf_token
        self.add_doc_boundaries  = add_doc_boundaries
        self.trust_remote_code   = trust_remote_code
        self.rank                = rank
        self.world_size          = world_size
        self.tokenizer           = ByteTokenizer()

    def _load_hf_dataset(self):
        kwargs = {}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        if self.trust_remote_code:
            kwargs["trust_remote_code"] = True
        ds = load_dataset(
            self.dataset_name,
            self.dataset_config_name,
            split=self.split,
            streaming=self.streaming,
            **kwargs,
        )
        if self.streaming:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)
        else:
            ds = ds.shuffle(seed=self.seed)
        if self.world_size > 1:
            ds = ds.shard(num_shards=self.world_size, index=self.rank)
        return ds

    def __iter__(self) -> Iterator[np.ndarray]:
        worker_info = torch.utils.data.get_worker_info()
        ds = self._load_hf_dataset()
        if worker_info is not None:
            ds = ds.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        buffer    = np.empty(0, dtype=np.uint8)
        block_len = self.seq_len + 1

        for example in ds:
            text = example.get(self.text_column) if isinstance(example, dict) else None
            if not text:
                continue
            encoded = self.tokenizer.encode(
                [text],
                add_bos=self.add_doc_boundaries,
                add_eos=self.add_doc_boundaries,
            )[0]["input_ids"]
            buffer = np.concatenate([buffer, encoded])
            while len(buffer) >= block_len:
                chunk  = buffer[:block_len]
                buffer = buffer[block_len:]
                yield chunk


# --------------------------------------------------------------------------- #
# Métricas do paper
# --------------------------------------------------------------------------- #
def compute_bpb(avg_nll_nats: float, bytes_per_token: float = 1.0) -> float:
    """BPB = NLL_nats / ln(2) / bytes_per_token.
    Para byte-level: bytes_per_token=1.0.
    Para comparar na escala BPE: bytes_per_token=4.6 (GPT-2/FineWeb-Edu).
    """
    return avg_nll_nats / math.log(2) / bytes_per_token


def compute_compression_ratio(boundary_indicators: torch.Tensor) -> float:
    """Lˢ⁺¹/Lˢ — fração de posições marcadas como boundary."""
    return boundary_indicators.float().mean().item()


def compute_ratio_loss(
    boundary_probs: torch.Tensor,
    boundary_indicators: torch.Tensor,
    N: float,
) -> torch.Tensor:
    """Equação 10 do paper — regulariza a taxa de compressão em direção a 1/N.
    F (não diferenciável) guia a direção; G (diferenciável) recebe o gradiente.
    """
    F_val = boundary_indicators.float().mean()   # stop-gradient implícito
    G_val = boundary_probs.mean()
    return (N / (N - 1)) * ((N - 1) * F_val * G_val + (1 - F_val) * (1 - G_val))


def compute_bpic(L0: int, boundary_ind_list: list) -> float:
    """BPIC = L0 / Lˢ estimado pela composição dos ratios de cada estágio.
    Mede quantos bytes brutos correspondem a cada chunk no estágio mais interno.
    Valor esperado: ~4.5–5 para 1-stage (similar ao GPT-2 tokenizer).
    """
    compound = 1.0
    for b in boundary_ind_list:
        compound *= b.float().mean().item()
    Ls = L0 * compound
    return L0 / Ls if Ls > 0 else float("inf")


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def make_csv_fields(n_stages: int = 1) -> list:
    base = [
        "step",
        "split",
        "wall_time",
        "lm_loss",
        "perplexity",
        "lb_loss",
        "ratio_loss",    # Lratio agregado (eq. 10)
        "total_loss",
        "bpb",           # bits-per-byte
        "bpic",          # bytes-per-innermost-chunk
    ]
    compression = [f"compression_L{s+1}/L{s}" for s in range(n_stages)]
    tail = ["lr", "tokens_per_sec"]
    return base + compression + tail


class CsvLogger:
    def __init__(self, path: str, n_stages: int = 1):
        self.path   = Path(path)
        self.fields = make_csv_fields(n_stages)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        self._file   = open(self.path, mode="a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fields)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def log(self, **kwargs):
        row = {k: kwargs.get(k, "") for k in self.fields}
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


def collate_fn(batch):
    batch     = np.stack(batch, axis=0)
    batch     = torch.from_numpy(batch.astype(np.int64))
    input_ids = batch[:, :-1].contiguous()
    targets   = batch[:, 1:].contiguous()
    return input_ids, targets


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
def build_model(model_config_path: str, device: str, dtype: torch.dtype):
    with open(model_config_path) as f:
        config = json.load(f)
    attn_cfg = AttnConfig(**config.pop("attn_cfg"))
    ssm_cfg  = SSMConfig(**config.pop("ssm_cfg"))
    hnet_cfg = HNetConfig(**config, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)
    model    = HNetForCausalLM(hnet_cfg, device=device, dtype=dtype)
    model.init_weights()
    return model, hnet_cfg


def count_boundary_stages(arch_layout) -> int:
    n      = 0
    layout = arch_layout
    while isinstance(layout, list) and len(layout) == 3:
        n     += 1
        layout = layout[1]
    return n


@torch.no_grad()
def evaluate(model, val_iter_factory, lb_n: list, device, eval_steps: int):
    """Validação: retorna (lm_loss, lb_loss, ratio_loss, bpb, bpic, stage_ratios)."""
    model.eval()
    data_iter = val_iter_factory()

    total_lm    = 0.0
    total_lb    = 0.0
    total_ratio = 0.0
    total_bpb   = 0.0
    total_bpic  = 0.0
    stage_ratio_sums: dict[str, float] = {}
    n_batches   = 0

    for _ in range(eval_steps):
        try:
            input_ids, targets = next(data_iter)
        except StopIteration:
            break

        input_ids = input_ids.to(device)
        targets   = targets.to(device)
        B, L0     = input_ids.shape

        output = model(input_ids)
        logits = output.logits

        lm_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
        )

        # lb_loss original (fallback)
        lb_loss = torch.zeros((), device=device)
        for bpred, n in zip(output.bpred_output, lb_n):
            lb_loss = lb_loss + load_balancing_loss(bpred, n)

        # ratio_loss (H-Net)
        ratio_loss = torch.zeros((), device=device)
        has_hnet = (
            hasattr(output, "boundary_probs_list") and
            hasattr(output, "boundary_ind_list")
        )
        if has_hnet and lb_n:
            for b_probs, b_inds, N in zip(
                output.boundary_probs_list, output.boundary_ind_list, lb_n
            ):
                ratio_loss = ratio_loss + compute_ratio_loss(b_probs, b_inds, N)

            for s, b_ind in enumerate(output.boundary_ind_list):
                key = f"compression_L{s+1}/L{s}"
                stage_ratio_sums[key] = (
                    stage_ratio_sums.get(key, 0.0) +
                    compute_compression_ratio(b_ind)
                )
            total_bpic += compute_bpic(L0, output.boundary_ind_list)

        total_lm    += lm_loss.item()
        total_lb    += lb_loss.item()
        total_ratio += ratio_loss.item()
        total_bpb   += compute_bpb(lm_loss.item())
        n_batches   += 1

    model.train()

    if n_batches == 0:
        return None, None, None, None, None, {}

    avg_ratios = {k: v / n_batches for k, v in stage_ratio_sums.items()}
    return (
        total_lm    / n_batches,   # lm_loss
        total_lb    / n_batches,   # lb_loss
        total_ratio / n_batches,   # ratio_loss
        total_bpb   / n_batches,   # bpb
        total_bpic  / n_batches,   # bpic
        avg_ratios,                # {"compression_L1/L0": ..., ...}
    )


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
    parser.add_argument("--max-steps",            type=int,   default=100_000)
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

    args = parser.parse_args()
    if args.dataset_config_name in (None, "None", ""):
        args.dataset_config_name = None

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

    if args.resume_from:
        if is_main:
            print(f"Retomando de {args.resume_from}")
        model.load_state_dict(torch.load(args.resume_from, map_location=device))

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parâmetros: {n_params/1e6:.1f}M")

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

    # otimizador
    param_groups = group_params(model)
    for g in param_groups:
        g.setdefault("weight_decay", args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    raw_model = model
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=args.ddp_find_unused_parameters)

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        cosine   = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return args.min_lr + (args.lr - args.min_lr) * cosine

    # ── acumuladores ────────────────────────────────────────────────────────
    model.train()
    step             = 0
    t0               = time.time()
    train_start      = t0
    steps_since_log  = 0

    running_lm       = 0.0
    running_lb       = 0.0
    running_ratio    = 0.0   # ratio_loss acumulado
    running_bpb      = 0.0   # bpb acumulado
    running_bpic     = 0.0   # bpic acumulado
    running_ratios: dict[str, float] = {}   # compression_L{s+1}/L{s}

    optimizer.zero_grad()
    data_iter = iter(loader)

    # ── loop principal ───────────────────────────────────────────────────────
    while step < args.max_steps:
        for _ in range(args.grad_accum_steps):
            try:
                input_ids, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                input_ids, targets = next(data_iter)

            input_ids = input_ids.to(device)
            targets   = targets.to(device)
            B, L0     = input_ids.shape

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
            ratio_loss = torch.zeros((), device=device)
            has_hnet   = (
                hasattr(output, "boundary_probs_list") and
                hasattr(output, "boundary_ind_list")
            )
            if has_hnet and lb_n:
                for b_probs, b_inds, N in zip(
                    output.boundary_probs_list, output.boundary_ind_list, lb_n
                ):
                    ratio_loss = ratio_loss + compute_ratio_loss(b_probs, b_inds, N)
                lb_loss = ratio_loss   # substitui o lb_loss genérico

            loss = lm_loss + args.load_balancing_weight * lb_loss
            (loss / args.grad_accum_steps).backward()

            # acumula (sem grad)
            with torch.no_grad():
                scale = 1.0 / args.grad_accum_steps
                running_lm    += lm_loss.item()    * scale
                running_lb    += lb_loss.item()    * scale
                running_ratio += ratio_loss.item() * scale
                running_bpb   += compute_bpb(lm_loss.item()) * scale

                if has_hnet and lb_n:
                    running_bpic += compute_bpic(L0, output.boundary_ind_list) * scale
                    for s, b_ind in enumerate(output.boundary_ind_list):
                        key = f"compression_L{s+1}/L{s}"
                        running_ratios[key] = (
                            running_ratios.get(key, 0.0) +
                            compute_compression_ratio(b_ind) * scale
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
                stats = torch.tensor(
                    [running_lm, running_lb, running_ratio, running_bpb, running_bpic],
                    device=device,
                )
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                stats /= world_size
                running_lm, running_lb, running_ratio, running_bpb, running_bpic = stats.tolist()

            if is_main:
                n            = steps_since_log
                avg_lm       = running_lm    / n
                avg_lb       = running_lb    / n
                avg_ratio    = running_ratio / n
                avg_bpb      = running_bpb   / n
                avg_bpic     = running_bpic  / n
                avg_ratios   = {k: v / n for k, v in running_ratios.items()}
                perplexity   = math.exp(min(avg_lm, 20))
                tok_per_step = args.batch_size * args.seq_len * args.grad_accum_steps * world_size
                tok_per_sec  = tok_per_step * n / max(dt, 1e-8)

                ratio_str = " | ".join(f"{k}={v:.3f}" for k, v in avg_ratios.items())
                print(
                    f"passo {step:6d} | lm {avg_lm:.4f} | ppl {perplexity:.2f}"
                    f" | bpb {avg_bpb:.4f} | bpic {avg_bpic:.2f}"
                    f" | ratio_loss {avg_ratio:.4f} | lb {avg_lb:.4f}"
                    + (f" | {ratio_str}" if ratio_str else "")
                    + f" | lr {lr:.2e} | {dt/n:.2f}s/step | {tok_per_sec:,.0f} tok/s"
                )
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
                    **avg_ratios,
                )

            # reset
            running_lm     = 0.0
            running_lb     = 0.0
            running_ratio  = 0.0
            running_bpb    = 0.0
            running_bpic   = 0.0
            running_ratios = {}
            steps_since_log = 0
            t0 = time.time()

        # ── validação ────────────────────────────────────────────────────────
        if is_main and val_loader and step > 0 and step % args.eval_every == 0:
            result = evaluate(
                raw_model, lambda: iter(val_loader), lb_n, device, args.eval_steps
            )
            val_lm, val_lb, val_ratio, val_bpb, val_bpic, val_stage_ratios = result

            if val_lm is not None:
                val_ppl = math.exp(min(val_lm, 20))
                ratio_str = " | ".join(f"{k}={v:.3f}" for k, v in val_stage_ratios.items())
                print(
                    f"          [val] passo {step:6d}"
                    f" | lm {val_lm:.4f} | ppl {val_ppl:.2f}"
                    f" | bpb {val_bpb:.4f} | bpic {val_bpic:.2f}"
                    f" | ratio_loss {val_ratio:.4f}"
                    + (f" | {ratio_str}" if ratio_str else "")
                )
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
                    **val_stage_ratios,
                )

        # ── checkpoint ───────────────────────────────────────────────────────
        if is_main and step > 0 and step % args.save_every == 0:
            ckpt = Path(args.out_dir) / f"step_{step}.pt"
            torch.save(raw_model.state_dict(), ckpt)
            print(f"Checkpoint salvo em {ckpt}")

    if is_main:
        final = Path(args.out_dir) / "final.pt"
        torch.save(raw_model.state_dict(), final)
        print(f"Treino concluído. Checkpoint final em {final}")
        csv_logger.close()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()