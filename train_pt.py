"""
train_pt.py

Script de treinamento do H-Net (Cartesia) para a língua portuguesa.

O dataset é carregado via `datasets.load_dataset` da Hugging Face, mas o
script foi feito para ser facilmente adaptável a QUALQUER corpus: basta
trocar os argumentos `--dataset-name`, `--dataset-config-name`,
`--dataset-split` e `--text-column` na linha de comando (ou os defaults
abaixo). Ele funciona com qualquer dataset que tenha uma coluna de texto
puro, streaming ou não.

Como o H-Net trabalha diretamente em bytes (ByteTokenizer), o script
concatena o texto de vários documentos num fluxo contínuo de bytes e o
corta em blocos de tamanho fixo (`--seq-len`), no estilo clássico de
treinamento de LM "packed" (sem padding) -- é exatamente o modo que
`HNetForCausalLM.forward` já otimiza quando `mask=None`.

Exemplo de uso com o CulturaX (português):

    python train_pt.py \
        --dataset-name uonlp/CulturaX \
        --dataset-config-name pt \
        --model-config configs/hnet_2stage_L.json \
        --seq-len 4096 \
        --batch-size 8

Trocando de corpus (ex.: OSCAR, mC4, ou um dataset local no formato HF):

    python train_pt.py \
        --dataset-name mc4 \
        --dataset-config-name pt \
        --text-column text \
        --model-config configs/hnet_2stage_L.json

Ou um dataset local já baixado:

    python train_pt.py \
        --dataset-name /caminho/para/dataset_local \
        --no-streaming \
        --model-config configs/hnet_2stage_L.json
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
import torch.nn.functional as F
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
# Dataset: concatena texto de QUALQUER dataset HF em um fluxo de bytes
# --------------------------------------------------------------------------- #
class ByteConcatDataset(IterableDataset):
    """
    Lê um dataset do Hugging Face (streaming ou não), tokeniza cada exemplo
    em bytes (ByteTokenizer) e concatena tudo em um fluxo contínuo, que é
    cortado em blocos de tamanho fixo `seq_len + 1` (entrada + alvo, alvo
    deslocado em 1 posição).

    Isso é agnóstico ao dataset: só depende do nome da coluna de texto.
    Para adaptar a outro corpus, basta mudar `dataset_name`,
    `dataset_config_name` e `text_column`.
    """

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
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.split = split
        self.text_column = text_column
        self.seq_len = seq_len
        self.streaming = streaming
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.hf_token = hf_token
        self.add_doc_boundaries = add_doc_boundaries
        self.tokenizer = ByteTokenizer()

    def _load_hf_dataset(self):
        kwargs = {}
        if self.hf_token is not None:
            # `token` é o nome atual do kwarg; datasets antigos usam
            # `use_auth_token`, mas `token` funciona nas versões recentes.
            kwargs["token"] = self.hf_token

        ds = load_dataset(
            self.dataset_name,
            self.dataset_config_name,
            split=self.split,
            streaming=self.streaming,
            **kwargs,
        )
        if self.streaming:
            # IterableDataset.shuffle usa buffer de reservatório -> aceita buffer_size
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)
        else:
            # Dataset.shuffle (modo mapa) não tem buffer_size, embaralha o índice inteiro
            ds = ds.shuffle(seed=self.seed)
        return ds

    def __iter__(self) -> Iterator[np.ndarray]:
        worker_info = torch.utils.data.get_worker_info()
        ds = self._load_hf_dataset()

        # Se houver múltiplos workers no DataLoader, cada um pega um "shard"
        # diferente do stream para não repetir dados.
        if worker_info is not None:
            if self.streaming:
                ds = ds.shard(num_shards=worker_info.num_workers, index=worker_info.id)
            else:
                ds = ds.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        buffer = np.empty(0, dtype=np.uint8)
        block_len = self.seq_len + 1  # +1 para termos entrada e alvo deslocado em 1

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
                chunk = buffer[:block_len]
                buffer = buffer[block_len:]
                yield chunk


# --------------------------------------------------------------------------- #
# Logging em CSV (treino + validação)
# --------------------------------------------------------------------------- #
CSV_FIELDS = [
    "step",
    "split",              # "train" ou "val"
    "wall_time",          # segundos desde o início do script
    "lm_loss",
    "perplexity",
    "lb_loss",
    "ratio_loss",         # <-- NOVO: Lratio do paper (eq. 10)
    "total_loss",
    "bpb",                # <-- NOVO: bits-per-byte
    "bpic",               # <-- NOVO: bytes-per-innermost-chunk
    "compression_L1/L0",  # <-- NOVO: ratio do estágio 0 (sempre presente)
    "compression_L2/L1",  # <-- NOVO: ratio do estágio 1 (vazio se 1-stage)
    "lr",
    "tokens_per_sec",
]

class CsvLogger:
    """Escreve métricas de treino/validação em CSV, uma linha por evento.

    O arquivo é aberto em modo 'append', então também funciona ao retomar
    um treino (`--resume-from`): as novas linhas são adicionadas ao final.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        self._file = open(self.path, mode="a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def log(self, **kwargs):
        row = {k: kwargs.get(k, "") for k in CSV_FIELDS}
        self._writer.writerow(row)
        self._file.flush()  # flush imediato: métricas ficam visíveis mesmo se o treino cair

    def close(self):
        self._file.close()


def collate_fn(batch):
    batch = np.stack(batch, axis=0)  # (B, seq_len + 1)
    batch = torch.from_numpy(batch.astype(np.int64))
    input_ids = batch[:, :-1].contiguous()
    targets = batch[:, 1:].contiguous()
    return input_ids, targets


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
def build_model(model_config_path: str, device: str, dtype: torch.dtype):
    with open(model_config_path, "r") as f:
        config = json.load(f)

    attn_cfg = AttnConfig(**config.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config.pop("ssm_cfg"))
    hnet_cfg = HNetConfig(**config, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)

    model = HNetForCausalLM(hnet_cfg, device=device, dtype=dtype)
    model.init_weights()
    return model, hnet_cfg


@torch.no_grad()
def evaluate(model, val_iter_factory, lb_n, device, eval_steps: int):
    """Roda `eval_steps` batches de validação e retorna (lm_loss médio, lb_loss médio).

    `val_iter_factory` é uma função que retorna um novo iterador do
    DataLoader de validação (para reiniciar o stream a cada avaliação).
    """
    model.eval()
    data_iter = val_iter_factory()

    total_lm_loss = 0.0
    total_lb_loss = 0.0
    n_batches = 0

    for _ in range(eval_steps):
        try:
            input_ids, targets = next(data_iter)
        except StopIteration:
            break

        input_ids = input_ids.to(device)
        targets = targets.to(device)

        output = model(input_ids)
        logits = output.logits

        lm_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
        )

        lb_loss = torch.zeros((), device=device)
        for bpred, n in zip(output.bpred_output, lb_n):
            lb_loss = lb_loss + load_balancing_loss(bpred, n)

        total_lm_loss += lm_loss.item()
        total_lb_loss += lb_loss.item()
        n_batches += 1

    model.train()

    if n_batches == 0:
        return None, None
    return total_lm_loss / n_batches, total_lb_loss / n_batches


def count_boundary_stages(arch_layout) -> int:
    """Quantos módulos de roteamento existem (== len(bpred_output) no forward)."""
    n = 0
    layout = arch_layout
    while isinstance(layout, list) and len(layout) == 3:
        n += 1
        layout = layout[1]
    return n


# --------------------------------------------------------------------------- #
# Loop de treino
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Treinamento do H-Net em português (ou qualquer corpus do Hugging Face)"
    )

    # ---- dataset (tudo adaptável) ----
    parser.add_argument("--dataset-name", type=str, default="uonlp/CulturaX",
                         help="Nome do dataset no Hugging Face Hub (ou caminho local)")
    parser.add_argument("--dataset-config-name", type=str, default="pt",
                         help="Subconjunto/config do dataset (ex.: idioma). Use 'None' se não houver")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default=None,
                         help="Split de validação (ex.: 'validation'). Se omitido, não roda validação. "
                              "Usa o mesmo --dataset-name/--dataset-config-name do treino")
    parser.add_argument("--text-column", type=str, default="text",
                         help="Nome da coluna com o texto bruto")
    parser.add_argument("--streaming", dest="streaming", action="store_true", default=True,
                         help="Usa streaming do datasets (recomendado para corpora grandes, ex. CulturaX)")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")

    # ---- modelo ----
    parser.add_argument("--model-config", type=str, required=True,
                         help="JSON com attn_cfg, ssm_cfg, arch_layout, d_model, etc. (mesmo formato usado em generate.py)")
    parser.add_argument("--resume-from", type=str, default=None,
                         help="Checkpoint .pt para retomar o treino")

    # ---- treino ----
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr-multiplier", type=str, default=None,
                         help="Multiplicadores de LR por estágio, separados por vírgula (ex.: '3.0,1.7,0.9'). "
                              "Se omitido, usa 1.0 para todos os estágios")
    parser.add_argument("--load-balancing-n", type=str, default=None,
                         help="Fator N (taxa de compressão alvo) por estágio de roteamento, separado por vírgula. "
                              "Se omitido, usa 4.0 para cada estágio")
    parser.add_argument("--load-balancing-weight", type=float, default=0.03)

    # ---- infra ----
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                         choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=500,
                         help="A cada quantos passos rodar validação (só tem efeito com --val-split)")
    parser.add_argument("--eval-steps", type=int, default=50,
                         help="Quantos batches de validação usar em cada avaliação")
    parser.add_argument("--out-dir", type=str, default="checkpoints/pt-hnet")
    parser.add_argument("--csv-path", type=str, default=None,
                         help="Caminho do CSV de métricas. Default: <out-dir>/metrics.csv")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.dataset_config_name in (None, "None", ""):
        args.dataset_config_name = None

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    # ---------------- dataset ----------------
    dataset = ByteConcatDataset(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        split=args.dataset_split,
        text_column=args.text_column,
        seq_len=args.seq_len,
        streaming=args.streaming,
        seed=args.seed,
        hf_token=HF_TOKEN,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    val_loader = None
    if args.val_split is not None:
        val_dataset = ByteConcatDataset(
            dataset_name=args.dataset_name,
            dataset_config_name=args.dataset_config_name,
            split=args.val_split,
            text_column=args.text_column,
            seq_len=args.seq_len,
            streaming=args.streaming,
            seed=args.seed,
            hf_token=HF_TOKEN,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=0,  # validação: evita overhead de workers para poucos batches
        )

    # ---------------- CSV de métricas ----------------
    csv_path = args.csv_path or str(Path(args.out_dir) / "metrics.csv")
    csv_logger = CsvLogger(csv_path)
    print(f"Métricas serão salvas em {csv_path}")

    # ---------------- modelo ----------------
    print("Construindo modelo...")
    model, hnet_cfg = build_model(args.model_config, device=device, dtype=dtype)
    n_boundary_stages = count_boundary_stages(hnet_cfg.arch_layout)
    n_total_stages = n_boundary_stages + 1
    print(f"Hierarquia com {n_total_stages} estágio(s) ({n_boundary_stages} módulo(s) de roteamento)")

    if args.resume_from is not None:
        print(f"Retomando checkpoint de {args.resume_from}")
        state_dict = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(state_dict)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total de parâmetros: {n_params / 1e6:.1f}M")

    # Sempre aplicamos multiplicadores de LR (mesmo que só 1.0), pois é essa
    # chamada que inicializa o dicionário `_optim` em cada parâmetro, usado
    # depois por `group_params` para montar os grupos do otimizador.
    if args.lr_multiplier is not None:
        lr_mult = [float(x) for x in args.lr_multiplier.split(",")]
    else:
        lr_mult = [1.0] * n_total_stages
    assert len(lr_mult) == n_total_stages, (
        f"--lr-multiplier precisa ter {n_total_stages} valores (um por estágio da hierarquia), "
        f"recebeu {len(lr_mult)}"
    )
    model.apply_lr_multiplier(lr_mult)

    # N de load balancing por estágio de roteamento
    if args.load_balancing_n is not None:
        lb_n = [float(x) for x in args.load_balancing_n.split(",")]
    else:
        lb_n = [4.0] * n_boundary_stages
    assert len(lb_n) == n_boundary_stages, (
        f"--load-balancing-n precisa ter {n_boundary_stages} valores (um por estágio de roteamento), "
        f"recebeu {len(lb_n)}"
    )

  # ---------------- otimizador ----------------
    param_groups = group_params(model)
    for g in param_groups:
        g.setdefault("weight_decay", args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return args.min_lr + (args.lr - args.min_lr) * cosine

    # ---------------- helpers de métricas ----------------
    import math as _math

    def compute_bpb(avg_nll_nats: float, bytes_per_token: float = 1.0) -> float:
        """BPB = NLL_nats / ln(2) / bytes_per_token"""
        return avg_nll_nats / _math.log(2) / bytes_per_token

    def compute_compression_ratio(boundary_indicators: torch.Tensor) -> float:
        """Fração de posições marcadas como boundary — Lˢ⁺¹/Lˢ"""
        return boundary_indicators.float().mean().item()

    def compute_ratio_loss(
        boundary_probs: torch.Tensor,
        boundary_indicators: torch.Tensor,
        N: float,
    ) -> torch.Tensor:
        """Equação 10 do paper: regulariza a taxa de compressão em direção a 1/N"""
        F_val = boundary_indicators.float().mean()          # não diferenciável
        G_val = boundary_probs.mean()                       # diferenciável
        return (N / (N - 1)) * ((N - 1) * F_val * G_val + (1 - F_val) * (1 - G_val))

    def compute_bpic(L0: int, boundary_ind_list: list[torch.Tensor]) -> float:
        """BPIC = L0 / Lˢ estimado pela composição dos ratios"""
        compound = 1.0
        for b in boundary_ind_list:
            compound *= b.float().mean().item()
        Ls = L0 * compound
        return L0 / Ls if Ls > 0 else float("inf")

    # ---------------- loop de treino ----------------
    model.train()
    step = 0
    t0 = time.time()
    train_start = t0

    # acumuladores — agora incluem as novas métricas
    running_loss       = 0.0
    running_lb_loss    = 0.0
    running_ratio_loss = 0.0                          # <-- NOVO
    running_bpb        = 0.0                          # <-- NOVO
    running_bpic       = 0.0                          # <-- NOVO
    # dicionário para ratios por estágio (quantidade dinâmica de estágios)
    running_ratios: dict[str, float] = {}             # <-- NOVO

    optimizer.zero_grad()
    data_iter = iter(loader)

    # quantos estágios o modelo tem e qual o N alvo de cada um
    # ex: [6.0] para 1-stage, [3.0, 3.0] para 2-stage
    hnet_N_per_stage: list[float] = getattr(args, "hnet_n_per_stage", [])
    alpha_ratio: float = getattr(args, "load_balancing_weight", 0.03)  # mesmo alpha do paper
    bytes_per_gpt2_token: float = 4.6   # FineWeb-Edu com GPT-2 tokenizer

    while step < args.max_steps:
        for _ in range(args.grad_accum_steps):
            try:
                input_ids, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                input_ids, targets = next(data_iter)

            input_ids = input_ids.to(device)
            targets   = targets.to(device)

            output = model(input_ids)
            logits = output.logits

            # ── perda autorregressiva ──────────────────────────────────────────
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1),
            )

            # ── load-balancing / ratio loss ────────────────────────────────────
            lb_loss    = torch.zeros((), device=device)
            ratio_loss = torch.zeros((), device=device)

            # Caso o modelo exponha boundary_probs e boundary_indicators
            # (atributos adicionados ao output do H-Net)
            has_hnet_outputs = (
                hasattr(output, "boundary_probs_list") and
                hasattr(output, "boundary_ind_list")
            )

            if has_hnet_outputs and hnet_N_per_stage:
                # ── métricas de chunking (H-Net) ───────────────────────────────
                for s, (b_probs, b_inds, N) in enumerate(
                    zip(output.boundary_probs_list, output.boundary_ind_list, hnet_N_per_stage)
                ):
                    ratio_loss = ratio_loss + compute_ratio_loss(b_probs, b_inds, N)

                # ratio loss substitui / complementa o lb_loss genérico
                lb_loss = ratio_loss

            else:
                # fallback: load-balancing original (MoE ou similar)
                for bpred, n in zip(output.bpred_output, lb_n):
                    lb_loss = lb_loss + load_balancing_loss(bpred, n)

            loss = lm_loss + alpha_ratio * lb_loss
            (loss / args.grad_accum_steps).backward()

            # ── acumula métricas (sem grad) ────────────────────────────────────
            with torch.no_grad():
                running_loss       += lm_loss.item()    / args.grad_accum_steps
                running_lb_loss    += lb_loss.item()    / args.grad_accum_steps
                running_ratio_loss += ratio_loss.item() / args.grad_accum_steps

                # BPB do mini-batch atual
                batch_bpb = compute_bpb(lm_loss.item(), bytes_per_token=1.0)
                running_bpb += batch_bpb / args.grad_accum_steps

                # Compression ratios e BPIC (só se H-Net)
                if has_hnet_outputs and hnet_N_per_stage:
                    L0 = input_ids.shape[1]
                    running_bpic += (
                        compute_bpic(L0, output.boundary_ind_list) / args.grad_accum_steps
                    )
                    for s, b_ind in enumerate(output.boundary_ind_list):
                        key = f"L{s+1}/L{s}"
                        running_ratios[key] = running_ratios.get(key, 0.0) + (
                            compute_compression_ratio(b_ind) / args.grad_accum_steps
                        )

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        lr = lr_at(step)
        for g in optimizer.param_groups:
            mult = g.get("lr_multiplier", 1.0)
            g["lr"] = lr * mult

        optimizer.step()
        optimizer.zero_grad()

        # ── log ───────────────────────────────────────────────────────────────
        if step % args.log_every == 0:
            dt = time.time() - t0
            n  = max(1, args.log_every)

            avg_lm_loss    = running_loss       / n
            avg_lb_loss    = running_lb_loss    / n
            avg_ratio_loss = running_ratio_loss / n
            avg_bpb        = running_bpb        / n
            avg_bpic       = running_bpic       / n
            avg_ratios     = {k: v / n for k, v in running_ratios.items()}

            perplexity       = math.exp(min(avg_lm_loss, 20))
            tokens_per_step  = args.batch_size * args.seq_len * args.grad_accum_steps
            tokens_per_sec   = (tokens_per_step * n) / max(dt, 1e-8)

            # linha de log compacta
            ratio_str = " | ".join(f"{k}={v:.3f}" for k, v in avg_ratios.items())
            print(
                f"passo {step:6d} | loss_lm {avg_lm_loss:.4f} | ppl {perplexity:.2f} "
                f"| bpb {avg_bpb:.4f} | bpic {avg_bpic:.2f} "
                f"| ratio_loss {avg_ratio_loss:.4f} | lb_loss {avg_lb_loss:.4f} "
                + (f"| {ratio_str} " if ratio_str else "")
                + f"| lr {lr:.2e} | {dt/n:.2f}s/passo | {tokens_per_sec:,.0f} tok/s"
            )

            log_kwargs = dict(
                step=step,
                split="train",
                wall_time=round(time.time() - train_start, 2),
                lm_loss=avg_lm_loss,
                perplexity=perplexity,
                lb_loss=avg_lb_loss,
                ratio_loss=avg_ratio_loss,          # <-- NOVO
                total_loss=avg_lm_loss + alpha_ratio * avg_lb_loss,
                bpb=avg_bpb,                        # <-- NOVO
                bpic=avg_bpic,                      # <-- NOVO
                lr=lr,
                tokens_per_sec=round(tokens_per_sec, 1),
                **{f"compression_{k}": v for k, v in avg_ratios.items()},  # <-- NOVO
            )
            csv_logger.log(**log_kwargs)

            # reset acumuladores
            running_loss       = 0.0
            running_lb_loss    = 0.0
            running_ratio_loss = 0.0
            running_bpb        = 0.0
            running_bpic       = 0.0
            running_ratios     = {}
            t0 = time.time()

        # ── validação ─────────────────────────────────────────────────────────
        if val_loader is not None and step > 0 and step % args.eval_every == 0:
            val_lm_loss, val_lb_loss = evaluate(
                model, lambda: iter(val_loader), lb_n, device, args.eval_steps
            )
            if val_lm_loss is not None and val_lb_loss is not None:
                val_ppl = math.exp(min(val_lm_loss, 20))
                val_bpb = compute_bpb(val_lm_loss, bytes_per_token=1.0)   # <-- NOVO

                print(
                    f"          [val] passo {step:6d} | loss_lm {val_lm_loss:.4f} "
                    f"| ppl {val_ppl:.2f} | bpb {val_bpb:.4f} "  # <-- NOVO
                    f"| loss_lb {val_lb_loss:.4f}"
                )
                csv_logger.log(
                    step=step,
                    split="val",
                    wall_time=round(time.time() - train_start, 2),
                    lm_loss=val_lm_loss,
                    perplexity=val_ppl,
                    lb_loss=val_lb_loss,
                    ratio_loss="",
                    total_loss=val_lm_loss + alpha_ratio * val_lb_loss,
                    bpb=val_bpb,                    # <-- NOVO
                    bpic="",
                    lr="",
                    tokens_per_sec="",
                )

        if step > 0 and step % args.save_every == 0:
            ckpt_path = Path(args.out_dir) / f"step_{step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint salvo em {ckpt_path}")

        step += 1

    final_path = Path(args.out_dir) / "final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Treino concluído. Checkpoint final em {final_path}")
    csv_logger.close()


if __name__ == "__main__":
    main()
