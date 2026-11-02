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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.transforms import Bbox
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


def place_labels(fig, ax, finals, marker_labels):
    """Posiciona rótulos sem sobreposição.

    Para cada rótulo testa várias posições ao redor do ponto e escolhe a que não
    colide com a legenda nem com rótulos já colocados e que cubra menos pontos das
    curvas. Valores finais têm prioridade (sempre exibidos); rótulos de marcadores
    que não encontrarem lugar livre são omitidos em vez de ficarem ilegíveis.
    """
    if not finals and not marker_labels:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axbox = ax.get_window_extent(renderer)

    placed = []
    legend = ax.get_legend()
    if legend is not None:
        placed.append(_padded_extent_box(legend.get_window_extent(renderer)))

    arrays = [ax.transData.transform(line.get_xydata())
              for line in ax.lines if len(line.get_xdata())]
    pts = np.vstack(arrays) if arrays else np.empty((0, 2))

    def points_inside(box):
        if not len(pts):
            return 0
        return int(((pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1) &
                    (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1)).sum())

    def draw(x, y, text, color, offset, final):
        dx, dy = offset
        ha = "left" if dx > 0 else "right" if dx < 0 else "center"
        va = "center" if dx != 0 else ("bottom" if dy > 0 else "top")
        return ax.annotate(
            text, (x, y), textcoords="offset points", xytext=offset,
            ha=ha, va=va, fontsize=8 if final else 7,
            fontweight="bold" if final else "normal", color=color, zorder=5,
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7, "alpha": 0.8,
                        "shrinkA": 0, "shrinkB": 3} if (abs(dy) > 10 or final) else None,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white",
                  "ec": color if final else "none", "lw": 0.8, "alpha": 0.92},
        )

    def anchored_box(anchor, size, offset, pad=4):
        """Caixa em pixels de um texto de tamanho `size` posto em `offset` (pontos) do ponto âncora."""
        scale = fig.dpi / 72.0
        dx, dy = offset[0] * scale, offset[1] * scale
        w, h = size
        cx, cy = anchor[0] + dx, anchor[1] + dy
        x0 = cx if offset[0] > 0 else cx - w if offset[0] < 0 else cx - w / 2
        y0 = cy - h / 2 if offset[0] != 0 else (cy if offset[1] > 0 else cy - h)
        return Bbox.from_extents(x0 - pad, y0 - pad, x0 + w + pad, y0 + h + pad)

    def place(x, y, text, color, offsets, final):
        probe = draw(x, y, text, color, (0, 8), final)
        ext = Text.get_window_extent(probe, renderer)
        probe.remove()
        # Margem extra para a moldura (bbox) desenhada ao redor do texto.
        size = (ext.width + 6, ext.height + 4)
        anchor = ax.transData.transform((x, y))

        best = None
        for offset in offsets:
            box = anchored_box(anchor, size, offset)
            inside_axes = (axbox.y0 <= box.y0 and box.y1 <= axbox.y1 and
                           (final or (axbox.x0 <= box.x0 and box.x1 <= axbox.x1)))
            if not inside_axes or any(box.overlaps(p) for p in placed):
                continue
            cost = points_inside(box)
            if best is None or cost < best[0]:
                best = (cost, offset, box)
            if cost == 0:
                break
        if best is None:
            return False
        _, offset, box = best
        draw(x, y, text, color, offset, final)
        placed.append(box)
        return True

    final_offsets = [(10, 0)] + [(10, s * 14 * k) for k in range(1, 8) for s in (1, -1)]
    marker_offsets = [(0, 8), (0, -8), (0, 20), (0, -20), (0, 32), (0, -32),
                      (10, 10), (10, -10), (-10, 10), (-10, -10)]

    for x, y, text, color in finals:
        place(x, y, text, color, final_offsets, final=True)
    for x, y, text, color in marker_labels:
        place(x, y, text, color, marker_offsets, final=False)


def _padded_extent_box(box, pad=3):
    return Bbox.from_extents(box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad)


def plot_metric(datasets, labels, metric: str, x_column: str, out_path: Path,
                trend_only: bool, trend_window: int, markers: int = 8,
                label_markers: bool = False, y_ticks: int = 15,
                annotate_final: bool = True):
    log_scale = metric in LOG_METRICS
    available = False
    fig, ax = plt.subplots(figsize=(11, 6.5))
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", [None])
    finals = []
    marker_labels = []

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
                        marker_labels.append((xs[i], ys[i], format_value(ys[i]), color))

            # Valor final em destaque (o texto é posicionado depois, sem colisão).
            if annotate_final:
                ax.plot([points[-1][0]], [final_y], marker="o" if is_train else "s",
                        markersize=8, color=color, markeredgecolor="black",
                        markeredgewidth=0.8, zorder=4)
                finals.append((points[-1][0], final_y, format_value(final_y), color))

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
    place_labels(fig, ax, finals, marker_labels)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
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