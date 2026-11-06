r"""
eval_pt.py  —  H-Net · avaliação de checkpoints (somente eval)

Percorre os checkpoints `step_*.pt` de uma pasta (ex.: checkpoints/train_pt/$hnet),
avalia cada um em um ou mais datasets e grava as MESMAS métricas do treinamento
(lm_loss, perplexity, lb_loss, ratio_loss, total_loss, bpb, bpic e as taxas de
compressão por estágio) em CSVs numa pasta própria — um CSV por dataset.

As métricas vêm de `train.evaluate.evaluate`, a mesma função usada na validação
do train_pt.py, então os números são diretamente comparáveis com os do treino.

Como funciona
  1. Os batches de avaliação de cada dataset são lidos UMA vez e ficam em memória
     (CPU): todos os checkpoints são avaliados exatamente nos mesmos dados.
  2. Cada checkpoint é carregado uma única vez e avaliado em todos os datasets.
  3. O CSV é regravado a cada (checkpoint, dataset) concluído. Rodar de novo só
     avalia o que falta (checkpoints novos, datasets novos, ou arquivo .pt
     modificado / --seq-len / --batch-size / --eval-steps diferentes).
     Use --overwrite para refazer tudo.

Uso (rode do mesmo jeito e do mesmo diretório que o train_pt.py)

  python eval_pt.py \
      --model-config configs/<config>.json \
      --checkpoint-dir checkpoints/train_pt/$hnet \
      --datasets-config eval_datasets.json \
      --csv-dir results/eval_pt/$hnet

  eval_datasets.json  (lista; só `dataset_name` é obrigatório):
    [
      {"label": "culturax_pt", "dataset_name": "uonlp/CulturaX", "config_name": "pt",
       "split": "train"},
      {"label": "outro_ds", "dataset_name": "org/dataset", "split": "validation",
       "text_column": "content", "eval_steps": 100}
    ]

  Campos: label, dataset_name, config_name, split (padrão "validation"),
          text_column (padrão "text"), streaming (padrão true),
          trust_remote_code (padrão false), eval_steps (padrão: --eval-steps).

  Ou direto na linha de comando (repetível, pares chave=valor):
    --dataset label=culturax_pt dataset_name=uonlp/CulturaX config_name=pt split=train \
    --dataset label=outro_ds dataset_name=org/dataset split=validation

Saída
  <csv-dir>/<label>.csv  — uma linha por checkpoint, ordenadas por step.
  Padrão de <csv-dir>: eval_results/<nome-da-pasta-do-checkpoint>.

Paralelizar em várias GPUs: um processo por grupo de datasets
  (CUDA_VISIBLE_DEVICES=0 ... --only ds_a  /  CUDA_VISIBLE_DEVICES=1 ... --only ds_b).
  Não rode dois processos escrevendo no MESMO CSV.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, fields
from itertools import islice
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from hnet.models.mixer_seq import load_from_checkpoint
from hnet.utils.arch import count_boundary_stages
from hnet.utils.datasets import ByteConcatDataset, collate_fn
from train.evaluate import evaluate

load_dotenv()
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
@dataclass
class DatasetSpec:
    label: str                          # nome curto: vira o nome do CSV e a coluna `dataset`
    dataset_name: str                   # id no HF Hub (ou o que o ByteConcatDataset aceitar)
    config_name: Optional[str] = None
    split: str = "validation"
    text_column: str = "text"
    streaming: bool = True
    trust_remote_code: bool = False
    eval_steps: Optional[int] = None    # None -> usa --eval-steps


_BOOL_KEYS = {"streaming", "trust_remote_code"}


def _is_none(v) -> bool:
    return v is None or str(v).strip().lower() in ("", "none", "null")


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "sim", "s"):
        return True
    if s in ("0", "false", "f", "no", "n", "nao", "não"):
        return False
    raise ValueError(f"valor booleano inválido: {v!r}")


def _sanitize(label: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", str(label)).strip("_") or "dataset"


def _spec_from_dict(d: dict, idx: int) -> DatasetSpec:
    valid = {f.name for f in fields(DatasetSpec)}
    unknown = set(d) - valid
    if unknown:
        raise ValueError(
            f"dataset #{idx}: chave(s) desconhecida(s) {sorted(unknown)}; válidas: {sorted(valid)}"
        )
    d = dict(d)
    if _is_none(d.get("dataset_name")):
        raise ValueError(f"dataset #{idx}: falta 'dataset_name'")
    for k in _BOOL_KEYS & d.keys():
        d[k] = _to_bool(d[k])
    if _is_none(d.get("eval_steps")):
        d.pop("eval_steps", None)
    else:
        d["eval_steps"] = int(d["eval_steps"])
    d["config_name"] = None if _is_none(d.get("config_name")) else str(d["config_name"])
    if _is_none(d.get("label")):
        parts = (d["dataset_name"], d["config_name"], d.get("split", "validation"))
        d["label"] = "__".join(str(p) for p in parts if p)
    d["label"] = _sanitize(d["label"])
    return DatasetSpec(**d)


def load_dataset_specs(args) -> list[DatasetSpec]:
    raw: list[dict] = []
    if args.datasets_config:
        with open(args.datasets_config, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):                  # aceita também {"datasets": [...]}
            data = data.get("datasets", [])
        raw.extend(data)
    for tokens in args.dataset or []:
        d = {}
        for tok in tokens:
            if "=" not in tok:
                raise ValueError(f"--dataset espera pares chave=valor, recebi {tok!r}")
            k, v = tok.split("=", 1)
            d[k.strip()] = v.strip()
        raw.append(d)
    if not raw:
        raise ValueError("Nenhum dataset informado: use --datasets-config e/ou --dataset")

    specs = [_spec_from_dict(d, i) for i, d in enumerate(raw)]
    labels = [s.label for s in specs]
    dup = sorted({x for x in labels if labels.count(x) > 1})
    if dup:
        raise ValueError(f"labels de dataset duplicados: {dup}")

    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        missing = sorted(wanted - set(labels))
        if missing:
            raise ValueError(f"--only tem labels inexistentes: {missing}. Disponíveis: {labels}")
        specs = [s for s in specs if s.label in wanted]
    return specs


def build_eval_batches(spec: DatasetSpec, args) -> list[tuple]:
    """Lê `eval_steps` batches do dataset e os mantém em CPU, para que todos os
    checkpoints sejam avaliados exatamente nos mesmos dados (e o stream seja lido
    uma única vez)."""
    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate_fn,
        "num_workers": args.num_workers,
    }

    if args.num_workers > 0:
        loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 2,
        })

    dataset = ByteConcatDataset(
        dataset_name=spec.dataset_name,
        dataset_config_name=spec.config_name,
        split=spec.split,
        text_column=spec.text_column,
        seq_len=args.seq_len,
        streaming=spec.streaming,
        seed=args.seed,
        hf_token=HF_TOKEN,
        trust_remote_code=spec.trust_remote_code,
    )
    loader = DataLoader(dataset, **loader_kwargs)
    n_steps = spec.eval_steps or args.eval_steps
    batches = [(x.cpu(), y.cpu()) for x, y in islice(loader, n_steps)]
    if not batches:
        raise RuntimeError(f"dataset '{spec.label}' não produziu nenhum batch")
    return batches


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
_STEP_RE = re.compile(r"^step_(\d+)\.pt$")


@dataclass
class Ckpt:
    path: Path
    step: Optional[int]     # None em final.pt (o passo real vem de dentro do .pt)
    mtime: int              # detecta checkpoint sobrescrito desde a última avaliação


def discover_checkpoints(args) -> list[Ckpt]:
    p = Path(args.checkpoint_dir)
    if p.is_file():                                  # um único .pt
        m = _STEP_RE.match(p.name)
        return [Ckpt(p, int(m.group(1)) if m else None, int(p.stat().st_mtime))]
    if not p.is_dir():
        raise FileNotFoundError(f"--checkpoint-dir não existe: {p}")

    out = []
    for f in p.glob("step_*.pt"):
        m = _STEP_RE.match(f.name)
        if not m:
            continue
        step = int(m.group(1))
        if args.min_step is not None and step < args.min_step:
            continue
        if args.max_step is not None and step > args.max_step:
            continue
        if args.every and step % args.every != 0:
            continue
        out.append(Ckpt(f, step, int(f.stat().st_mtime)))
    out.sort(key=lambda c: c.step)

    final = p / "final.pt"
    if args.include_final and final.is_file():
        out.append(Ckpt(final, None, int(final.stat().st_mtime)))
    return out


# --------------------------------------------------------------------------- #
# CSV (um por dataset)
# --------------------------------------------------------------------------- #
# Colunas fixas; as taxas de compressão por estágio (compression_L1/L0, ...) vêm
# depois, na ordem em que aparecem. Os nomes seguem o CSV do treino.
BASE_FIELDS = [
    "step", 
    "checkpoint", 
    "dataset", 
    "split",
    "lm_loss", 
    "perplexity", 
    "lb_loss", 
    "ratio_loss", 
    "total_loss", 
    "bpb", 
    "bpic",
    "tokens",        # total_tokens salvo dentro do checkpoint (contado pelo train_pt.py)
    "eval_tokens",   # tokens (bytes) usados nesta avaliação
    "seq_len", 
    "batch_size", 
    "eval_steps",
    "eval_time_s",
    "ckpt_mtime",    # mtime do .pt no momento da avaliação (controle de re-execução)
]


def _step_key(row: dict):
    name = str(row.get("checkpoint", ""))
    try:
        return (float(row.get("step", "")), name)
    except (TypeError, ValueError):
        return (float("inf"), name)


class DatasetCsv:
    """Regrava o arquivo inteiro (de forma atômica) a cada linha nova. Assim dá para
    acrescentar colunas, manter a ordem por step e substituir linhas reavaliadas."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict] = []
        self.extra_cols: list[str] = []
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.rows = list(reader)
                self.extra_cols = [c for c in (reader.fieldnames or []) if c not in BASE_FIELDS]

    def is_done(self, checkpoint: str, signature: dict) -> bool:
        return any(
            r.get("checkpoint") == checkpoint
            and all(str(r.get(k, v)) == str(v) for k, v in signature.items())
            for r in self.rows
        )

    def add(self, row: dict) -> None:
        for k in row:
            if k not in BASE_FIELDS and k not in self.extra_cols:
                self.extra_cols.append(k)
        self.rows = [r for r in self.rows if r.get("checkpoint") != row["checkpoint"]]
        self.rows.append(row)
        self.rows.sort(key=_step_key)
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=BASE_FIELDS + self.extra_cols,
                               restval="", extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(self.rows)
        os.replace(tmp, self.path)


# --------------------------------------------------------------------------- #
# Avaliação
# --------------------------------------------------------------------------- #
def parse_lb_n(spec: Optional[str], n_stages: int) -> list[float]:
    vals = [float(x) for x in spec.split(",")] if spec else [4.0] * n_stages
    if len(vals) != n_stages:
        # erro de configuração (igual para todos os checkpoints): aborta de vez.
        # SystemExit não é capturado pelo `except Exception` do loop de checkpoints.
        raise SystemExit(
            f"Erro: --load-balancing-n tem {len(vals)} valor(es), mas o modelo tem "
            f"{n_stages} estágio(s) de roteamento"
        )
    return vals


def eval_pair(model, spec: DatasetSpec, batches, lb_n, args, device) -> Optional[dict]:
    """Avalia o modelo em um dataset. Devolve a linha do CSV (sem step/checkpoint)."""
    n_steps = spec.eval_steps or args.eval_steps
    model.eval()        # evaluate() pode devolver o modelo em modo train; garante eval
    t0 = time.time()
    with torch.no_grad():
        result = evaluate(model, lambda: iter(batches), lb_n, device, n_steps)
    dt = time.time() - t0

    if result is None or result[0] is None:
        return None
    lm, lb, ratio, bpb, bpic, stage_ratios = result
    lm, lb = float(lm), float(lb)

    row = {
        "dataset": spec.label,
        "split": spec.split,
        "lm_loss": lm,
        "perplexity": math.exp(min(lm, 20)),
        "lb_loss": lb,
        "ratio_loss": float(ratio),
        "total_loss": lm + args.load_balancing_weight * lb,
        "bpb": float(bpb),
        "bpic": float(bpic),
        "eval_tokens": sum(x.numel() for x, _ in batches[:n_steps]),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "eval_steps": n_steps,
        "eval_time_s": round(dt, 2),
    }
    for k, v in (stage_ratios or {}).items():
        row[k] = float(v)
    return row


def format_row(row: dict) -> str:
    ratio_str = " | ".join(
        f"{k}={row[k]:.3f}" for k in row if k not in BASE_FIELDS
    )
    return (
        f"  [{row['dataset']}] lm {row['lm_loss']:.4f} | ppl {row['perplexity']:.2f}"
        f" | bpb {row['bpb']:.4f} | bpic {row['bpic']:.2f}"
        f" | ratio_loss {row['ratio_loss']:.4f} | lb {row['lb_loss']:.4f}"
        + (f" | {ratio_str}" if ratio_str else "")
        + f" | {row['eval_time_s']:.1f}s"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="H-Net: avalia checkpoints (somente eval).")

    # modelo / checkpoints
    p.add_argument("--model-config", type=str, required=True,
                   help="mesmo --model-config usado no treino")
    p.add_argument("--checkpoint-dir", type=str, required=True,
                   help="pasta com step_*.pt (ex.: checkpoints/train_pt/$hnet) ou um único .pt")
    p.add_argument("--min-step", type=int, default=None)
    p.add_argument("--max-step", type=int, default=None)
    p.add_argument("--every", type=int, default=None,
                   help="avalia só steps múltiplos de N (ex.: 5000)")
    p.add_argument("--include-final", action="store_true", default=False,
                   help="inclui também final.pt")

    # datasets
    p.add_argument("--datasets-config", type=str, default=None,
                   help="JSON com a lista de datasets (veja o topo do arquivo)")
    p.add_argument("--dataset", action="append", nargs="+", metavar="CHAVE=VALOR", default=None,
                   help="dataset inline, ex.: label=x dataset_name=org/ds split=validation "
                        "(repetível)")
    p.add_argument("--only", type=str, default=None,
                   help="labels (separados por vírgula) a avaliar; os demais são ignorados")

    # avaliação (mesmos padrões do treino)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--load-balancing-n", type=str, default=None,
                   help="N por estágio de roteamento, ex.: '4,4' (padrão: 4 em cada estágio)")
    p.add_argument("--load-balancing-weight", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", type=str, default=None,
                   help="padrão: cuda se disponível, senão cpu")

    # saída
    p.add_argument("--csv-dir", type=str, default=None,
                   help="pasta dos CSVs (padrão: eval_results/<nome da pasta do checkpoint>)")
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="reavalia mesmo o que já está nos CSVs")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="só mostra o que seria avaliado")

    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="número de processos CPU usados para carregar os dados (padrão: 0)"
    )
    
    return p.parse_args()


def main() -> int:
    args = parse_args()
    specs = load_dataset_specs(args)
    ckpts = discover_checkpoints(args)
    if not ckpts:
        log(f"Nenhum checkpoint encontrado em {args.checkpoint_dir} "
            f"(padrão step_*.pt, com os filtros informados).")
        return 1

    src = Path(args.checkpoint_dir)
    src_dir = src if src.is_dir() else src.parent
    csv_dir = Path(args.csv_dir) if args.csv_dir else Path("eval_results") / src_dir.resolve().name
    tables = {s.label: DatasetCsv(csv_dir / f"{s.label}.csv") for s in specs}

    def signature(spec: DatasetSpec, ck: Ckpt) -> dict:
        return {
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "eval_steps": spec.eval_steps or args.eval_steps,
            "ckpt_mtime": ck.mtime,
        }

    # plano: (checkpoint, datasets que ainda faltam)
    plan, n_skipped = [], 0
    for ck in ckpts:
        todo = [s for s in specs
                if args.overwrite or not tables[s.label].is_done(ck.path.name, signature(s, ck))]
        n_skipped += len(specs) - len(todo)
        if todo:
            plan.append((ck, todo))
    n_runs = sum(len(t) for _, t in plan)

    log(f"{len(ckpts)} checkpoint(s) x {len(specs)} dataset(s) | "
        f"a avaliar: {n_runs} | já nos CSVs (pulados): {n_skipped}")
    log(f"CSVs em: {csv_dir}")
    if args.dry_run:
        for ck, todo in plan:
            log(f"  {ck.path.name}: {', '.join(s.label for s in todo)}")
        return 0
    if not plan:
        log("Nada a avaliar. Use --overwrite para refazer.")
        return 0

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)
    log(f"device={device} | dtype={args.dtype} | seq_len={args.seq_len} | "
        f"batch_size={args.batch_size} | eval_steps={args.eval_steps}")

    # batches de avaliação: lidos uma vez, só dos datasets que ainda têm pendência
    needed = {s.label for _, todo in plan for s in todo}
    eval_batches: dict[str, list] = {}
    for s in specs:
        if s.label not in needed:
            continue
        cfg = f"/{s.config_name}" if s.config_name else ""
        log(f"[dados] {s.label}: {s.dataset_name}{cfg} [{s.split}]")
        eval_batches[s.label] = build_eval_batches(s, args)
        n_tok = sum(x.numel() for x, _ in eval_batches[s.label])
        log(f"        {len(eval_batches[s.label])} batch(es) | {n_tok:,} tokens")

    n_ok, failed = 0, []
    t_start = time.time()
    for i, (ck, todo) in enumerate(plan, 1):
        log(f"\n=== [{i}/{len(plan)}] {ck.path.name} ===")
        model = None
        try:
            model, hnet_cfg, meta = load_from_checkpoint(
                model_config_path=args.model_config,
                checkpoint_path=str(ck.path),
                device=device,
                dtype=dtype,
            )
            n_boundary = count_boundary_stages(hnet_cfg.arch_layout)
            step = meta.get("step")
            tokens = meta.get("total_tokens")
            del hnet_cfg, meta
            if step is None:
                step = ck.step
            lb_n = parse_lb_n(args.load_balancing_n, n_boundary)

            tok_str = f" | {tokens / 1e9:.2f}B tokens de treino" if tokens else ""
            log(f"  passo {step}{tok_str}")

            for s in todo:
                row = eval_pair(model, s, eval_batches[s.label], lb_n, args, device)
                if row is None:
                    log(f"  [{s.label}] evaluate() não retornou dados — ignorado")
                    continue
                row.update(step=step, checkpoint=ck.path.name,
                           tokens=tokens, ckpt_mtime=ck.mtime)
                tables[s.label].add(row)
                n_ok += 1
                log(format_row(row))
        except Exception:
            failed.append(ck.path.name)
            log(f"  ERRO em {ck.path.name} (seguindo para o próximo):")
            traceback.print_exc()
        finally:
            model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    log(f"\nConcluído em {time.time() - t_start:.0f}s | {n_ok} avaliação(ões) gravada(s) em {csv_dir}")
    if failed:
        log(f"{len(failed)} checkpoint(s) com erro: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("\nInterrompido. O que já foi avaliado está nos CSVs; rode de novo para continuar.")
        sys.exit(130)