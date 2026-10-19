""" 
plot_metrics_merged.py

Compara métricas de treinamento de múltiplos CSVs em gráficos combinados.

Exemplos:
    python plot_metrics_merged.py \
        --csv runs/model_a/metrics.csv runs/model_b/metrics.csv \
        --out-dir plots/merged

    python plot_metrics_merged.py \
        --csv runs/model_a/metrics.csv runs/model_b/metrics.csv \
        --x-column tokens --trend-only --trend-window 15

--trend-only plota somente tendências suavizadas por média móvel centrada.
O tamanho da janela é contado em pontos de cada série, não em unidades do eixo X.
"""

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


PREFERRED_METRICS = [
    "lm_loss",
    "perplexity",
    "lb_loss",
    "ratio_loss",
    "total_loss",
    "bpb",
    "bpic",
    "lr",
    "tokens_per_sec",
    "wall_time",
    "tokens",
]

LOG_METRICS = {"perplexity"}

LABELS = {
    "lm_loss": "Loss (LM, cross-entropy)",
    "perplexity": "Perplexity",
    "lb_loss": "Load balancing loss",
    "ratio_loss": "Ratio loss",
    "total_loss": "Loss total",
    "bpb": "Bits per byte (BPB)",
    "bpic": "Bytes per innermost chunk (BPIC)",
    "lr": "Learning rate",
    "tokens_per_sec": "Tokens por segundo",
    "wall_time": "Tempo decorrido (s)",
    "tokens": "Tokens processados",
}


def parse_number(value):
    """Converte para float; retorna None para vazios ou valores inválidos."""
    if value is None:
        return None

    value = str(value).strip()
    if not value:
        return None

    try:
        number = float(value)
    except (ValueError, TypeError):
        return None

    return number if math.isfinite(number) else None


def load_metrics(csv_path: Path):
    """Carrega métricas numéricas e separa as linhas de treino e validação."""
    train_rows = []
    val_rows = []

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []

        for row in reader:
            step = parse_number(row.get("step"))
            if step is None:
                continue

            parsed = {"step": step}
            parsed["split"] = (row.get("split") or "").strip().lower()

            for field in fields:
                if field not in ("step", "split"):
                    parsed[field] = parse_number(row.get(field))

            if parsed["split"] == "train":
                train_rows.append(parsed)
            elif parsed["split"] in ("val", "valid", "validation"):
                val_rows.append(parsed)

    return train_rows, val_rows, fields


def source_label(path: Path) -> str:
    """Retorna o nome-base do arquivo sem diretório nem extensões."""
    name = path.name
    suffixes = path.suffixes
    for suffix in reversed(suffixes):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name or path.stem or path.name


def unique_labels(paths):
    """Garante legendas diferentes quando arquivos têm o mesmo nome-base."""
    labels = [source_label(path) for path in paths]
    totals = {label: labels.count(label) for label in set(labels)}

    result = []
    for path, label in zip(paths, labels):
        if totals[label] > 1:
            label = f"{path.parent.name}/{label}"
        result.append(label)

    # Segundo desempate caso até a pasta-pai também seja igual.
    seen = {}
    for i, label in enumerate(result):
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            result[i] = f"{label} ({seen[label]})"
    return result


def metric_label(key: str) -> str:
    if key in LABELS:
        return LABELS[key]
    return key.replace("_", " ").replace("/", " / ")


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_.") or "metric"


def get_points(rows, x_column: str, metric: str, log_scale: bool):
    """Extrai pontos válidos e os ordena pelo eixo X."""
    points = []
    for row in rows:
        x = parse_number(row.get(x_column))
        y = parse_number(row.get(metric))
        if x is None or y is None:
            continue
        if log_scale and y <= 0:
            continue
        points.append((x, y))

    points.sort(key=lambda point: point[0])
    return points


def moving_average(points, window: int):
    """Calcula média móvel centrada, reduzindo a janela nas extremidades."""
    if not points or window <= 1:
        return points

    values = [point[1] for point in points]
    n = len(values)
    radius = window // 2
    smoothed = []

    for i, (x, _) in enumerate(points):
        start = max(0, i - radius)
        end = min(n, i + radius + 1)
        smoothed.append((x, sum(values[start:end]) / (end - start)))

    return smoothed


def plot_metric(datasets, labels, metric: str, x_column: str, out_path: Path,
                trend_only: bool, trend_window: int):
    log_scale = metric in LOG_METRICS
    available = False
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", [None])

    for index, dataset in enumerate(datasets):
        color = palette[index % len(palette)]
        label = labels[index]

        train_points = get_points(dataset["train"], x_column, metric, log_scale)
        val_points = get_points(dataset["val"], x_column, metric, log_scale)

        for split_name, points in (("Treino", train_points), ("Validação", val_points)):
            if not points:
                continue

            available = True
            shown_points = moving_average(points, trend_window) if trend_only else points
            xs, ys = zip(*shown_points)

            if trend_only:
                line_label = label if split_name == "Treino" else f"{label} (val)"
                ax.plot(xs, ys, label=line_label, color=color, linewidth=2.2,
                        linestyle="-" if split_name == "Treino" else "--")
            else:
                line_label = label if split_name == "Treino" else f"{label} (val)"
                ax.plot(xs, ys, label=line_label, color=color,
                        alpha=0.85 if split_name == "Treino" else 0.95,
                        linewidth=1.5 if split_name == "Treino" else 1.8,
                        linestyle="-" if split_name == "Treino" else "--")

    if not available:
        plt.close(fig)
        return False

    label = metric_label(metric)
    ax.set_xlabel(metric_label(x_column) if x_column in LABELS else x_column.replace("_", " "))
    ax.set_ylabel(label)
    ax.set_title(f"{label} por {ax.get_xlabel()}" + (" — tendência" if trend_only else ""))

    if log_scale:
        ax.set_yscale("log")

    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Salvo: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Gera gráficos comparativos a partir de múltiplos CSVs do H-Net."
    )
    parser.add_argument(
        "--csv", nargs="+", required=True, type=Path, metavar="ARQUIVO.csv",
        help="Um ou mais CSVs para comparar.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("plots/merged"),
        help="Diretório de saída (padrão: plots/merged).",
    )
    parser.add_argument(
        "--x-column", default="step",
        help="Coluna usada no eixo X (padrão: step; por exemplo: tokens).",
    )
    parser.add_argument(
        "--trend-only", action="store_true",
        help="Mostra somente tendências suavizadas por média móvel.",
    )
    parser.add_argument(
        "--trend-window", type=int, default=15,
        help="Janela da média móvel em pontos (padrão: 15; use 1 para sem suavização).",
    )
    args = parser.parse_args()

    if args.trend_window < 1:
        parser.error("--trend-window deve ser maior ou igual a 1.")

    paths = args.csv
    for path in paths:
        if not path.is_file():
            parser.error(f"CSV não encontrado: {path}")

    labels = unique_labels(paths)
    datasets = []
    all_fields = set()

    for path, label in zip(paths, labels):
        train_rows, val_rows, fields = load_metrics(path)
        all_fields.update(fields)
        datasets.append({"path": path, "label": label, "train": train_rows, "val": val_rows})
        print(f"Carregado: {path} ({len(train_rows)} treino, {len(val_rows)} validação) -> {label}")

    if args.x_column not in all_fields:
        parser.error(
            f"A coluna X '{args.x_column}' não existe nos CSVs fornecidos. "
            f"Colunas encontradas: {', '.join(sorted(all_fields))}"
        )

    # Métricas padrão + quaisquer métricas compression_* encontradas nos CSVs.
    compression_metrics = sorted(
        field for field in all_fields if field.startswith("compression_")
    )
    metrics = [
        metric for metric in PREFERRED_METRICS + compression_metrics
        if metric in all_fields and metric != args.x_column
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    generated = 0

    for metric in metrics:
        out_path = args.out_dir / f"{safe_filename(metric)}.png"
        if plot_metric(
            datasets, labels, metric, args.x_column, out_path,
            trend_only=args.trend_only, trend_window=args.trend_window,
        ):
            generated += 1

    if generated == 0:
        print("Nenhum gráfico foi gerado: não há métricas numéricas válidas em comum com os dados carregados.")
    else:
        mode = "tendências" if args.trend_only else "séries completas"
        print(f"\nConcluído: {generated} gráfico(s) ({mode}) em {args.out_dir}")


if __name__ == "__main__":
    main()
