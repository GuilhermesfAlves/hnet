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
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MaxNLocator


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

# Métricas determinísticas: suavizar só distorce o valor (principalmente o final).
NO_SMOOTH_METRICS = {"lr", "tokens", "wall_time"}

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


def format_value(value: float) -> str:
    """Formata o valor com 4 algarismos significativos (notação científica se muito pequeno/grande)."""
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude < 1e-3 or magnitude >= 1e5:
        return f"{value:.3e}"
    return f"{value:.4g}"


def marker_indices(n: int, count: int):
    """Índices igualmente espaçados (incluindo primeiro e último) para N marcadores."""
    if n <= 0 or count <= 0:
        return []
    if count == 1 or n == 1:
        return [n - 1]
    count = min(count, n)
    return sorted({round(i * (n - 1) / (count - 1)) for i in range(count)})


def plot_metric(datasets, labels, metric: str, x_column: str, out_path: Path,
                trend_only: bool, trend_window: int, markers: int = 8,
                label_markers: bool = False, y_ticks: int = 15,
                annotate_final: bool = True):
    log_scale = metric in LOG_METRICS
    available = False
    fig, ax = plt.subplots(figsize=(11, 6.5))
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", [None])
    final_count = 0

    for index, dataset in enumerate(datasets):
        color = palette[index % len(palette)]
        label = labels[index]

        train_points = get_points(dataset["train"], x_column, metric, log_scale)
        val_points = get_points(dataset["val"], x_column, metric, log_scale)

        for split_name, points in (("Treino", train_points), ("Validação", val_points)):
            if not points:
                continue

            available = True
            is_train = split_name == "Treino"
            smooth = trend_only and metric not in NO_SMOOTH_METRICS
            shown_points = moving_average(points, trend_window) if smooth else points
            xs, ys = zip(*shown_points)
            line_label = label if is_train else f"{label} (val)"
            final_y = points[-1][1]  # sempre o valor bruto final, nunca o suavizado

            # Legenda já traz o valor final, para leitura exata sem depender do eixo.
            legend_label = f"{line_label} — final: {format_value(final_y)}"

            if trend_only:
                ax.plot(xs, ys, label=legend_label, color=color, linewidth=2.2,
                        linestyle="-" if is_train else "--")
            else:
                ax.plot(xs, ys, label=legend_label, color=color,
                        alpha=0.85 if is_train else 0.95,
                        linewidth=1.5 if is_train else 1.8,
                        linestyle="-" if is_train else "--")

            # N marcadores ao longo da curva (opcionalmente com o valor escrito).
            idxs = marker_indices(len(xs), markers)
            if idxs:
                ax.plot([xs[i] for i in idxs], [ys[i] for i in idxs],
                        linestyle="none", marker="o" if is_train else "s",
                        markersize=4.5, color=color, zorder=3)
                if label_markers:
                    for i in idxs[:-1]:  # o último recebe a anotação do valor final
                        ax.annotate(format_value(ys[i]), (xs[i], ys[i]),
                                    textcoords="offset points", xytext=(0, 7),
                                    ha="center", fontsize=6.5, color=color)

            # Valor final em destaque, escalonado para evitar sobreposição.
            if annotate_final:
                text = format_value(final_y)
                ax.plot([points[-1][0]], [final_y], marker="o" if is_train else "s",
                        markersize=8, color=color, markeredgecolor="black",
                        markeredgewidth=0.8, zorder=4)
                ax.annotate(text, (points[-1][0], final_y), textcoords="offset points",
                            xytext=(10, (-1) ** final_count * 10 * (1 + final_count // 2)),
                            fontsize=8, color=color, fontweight="bold",
                            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8},
                            bbox={"boxstyle": "round,pad=0.2", "fc": "white",
                                  "ec": color, "lw": 0.8, "alpha": 0.9})
                final_count += 1

    if not available:
        plt.close(fig)
        return False

    label = metric_label(metric)
    ax.set_xlabel(metric_label(x_column) if x_column in LABELS else x_column.replace("_", " "))
    ax.set_ylabel(label)
    ax.set_title(f"{label} por {ax.get_xlabel()}" + (" — tendência" if trend_only else ""))

    if log_scale:
        ax.set_yscale("log")
    elif y_ticks > 0:
        # Mais marcações no eixo Y, com rótulos de precisão adequada.
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks, min_n_ticks=y_ticks // 2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: format_value(v)))
        ax.grid(which="minor", axis="y", alpha=0.12)

    if annotate_final:
        ax.margins(x=0.1)  # espaço à direita para as anotações finais

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
    parser.add_argument(
        "--markers", type=int, default=8, metavar="N",
        help="Número de marcadores igualmente espaçados em cada curva (padrão: 8; 0 desliga).",
    )
    parser.add_argument(
        "--label-markers", action="store_true",
        help="Escreve o valor de Y ao lado de cada marcador.",
    )
    parser.add_argument(
        "--y-ticks", type=int, default=15, metavar="N",
        help="Número aproximado de marcações no eixo Y (padrão: 15; 0 mantém o automático).",
    )
    parser.add_argument(
        "--no-final-annotation", action="store_true",
        help="Não anota o valor final de cada curva.",
    )
    args = parser.parse_args()

    if args.trend_window < 1:
        parser.error("--trend-window deve ser maior ou igual a 1.")
    if args.markers < 0 or args.y_ticks < 0:
        parser.error("--markers e --y-ticks devem ser maiores ou iguais a 0.")

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
            markers=args.markers, label_markers=args.label_markers,
            y_ticks=args.y_ticks, annotate_final=not args.no_final_annotation,
        ):
            generated += 1

    if generated == 0:
        print("Nenhum gráfico foi gerado: não há métricas numéricas válidas em comum com os dados carregados.")
    else:
        mode = "tendências" if args.trend_only else "séries completas"
        print(f"\nConcluído: {generated} gráfico(s) ({mode}) em {args.out_dir}")


if __name__ == "__main__":
    main()