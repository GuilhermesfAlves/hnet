""" 
plot_metrics_merged.py

Compara métricas de treinamento de múltiplos CSVs em gráficos combinados.
Aceita também CSVs gerados pelo eval_pt.py (uma linha por checkpoint): eles são
tratados como validação e usam a coluna total_tokens como eixo X "tokens".

Exemplos:
    python plot_metrics_merged.py \
        --csv runs/model_a/metrics.csv runs/model_b/metrics.csv \
        --out-dir plots/merged

    python plot_metrics_merged.py \
        --csv checkpoints/train_pt/hnet_1stage_L/metrics.csv \
              results/eval_pt/hnet_1stage_L/culturax_pt.csv \
        --x-column tokens --trend-only --trend-window 15 --eval-trend-window 1

Por padrão o eixo Y é ajustado ao intervalo central que contém 95% dos dados
(--focus-pct), para que a região de interesse ocupe o gráfico todo; o intervalo
exato aparece no subtítulo. Valores finais sempre ficam visíveis. O tamanho das
fontes é controlado por --font-size.

Saída: --format png (padrão), pdf, svg ou várias (ex.: --format png pdf).
Eixo X: valores divididos por 2^k (--x-exp, padrão 10, ou seja N × 2^10); o
fator aparece no rótulo do eixo e não há multiplicador tipo "1e9" no canto.

--trend-only plota somente tendências suavizadas por média móvel centrada.
O tamanho da janela é contado em pontos de cada série, não em unidades do eixo X.
--trend-window vale para curvas de treino; --eval-trend-window para as de
eval/validação (padrão 1 = sem suavização, pois há poucos pontos).

Cores: a curva de eval/validação usa a mesma cor do treino, só que mais clara.
"""

import argparse
import ast
import csv
import math
import operator
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator


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


def lighten(color, amount=0.45):
    """Clareia uma cor misturando-a com branco (amount: 0 = igual, 1 = branco)."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


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
    """Carrega métricas numéricas e separa as linhas de treino e validação.

    CSVs do eval_pt.py (têm as colunas 'checkpoint' e 'dataset') são tratados
    inteiramente como validação, e 'total_tokens' vira o eixo 'tokens'.
    """
    train_rows = []
    val_rows = []

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        is_eval = "checkpoint" in fields and "dataset" in fields

        for row in reader:
            step = parse_number(row.get("step"))
            if step is None:
                continue

            parsed = {"step": step}
            parsed["split"] = (row.get("split") or "").strip().lower()

            for field in fields:
                if field not in ("step", "split"):
                    parsed[field] = parse_number(row.get(field))

            # eval: usa o total_tokens do checkpoint como eixo X "tokens"
            # (ignora checkpoints salvos sem esse campo, com total_tokens = 0)
            if is_eval and parsed.get("total_tokens"):
                parsed["tokens"] = parsed["total_tokens"]

            if is_eval:
                val_rows.append(parsed)
            elif parsed["split"] == "train":
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


def get_points(rows, x_column: str, metric: str, log_scale: bool, x_div: float = 1.0):
    """Extrai pontos válidos e os ordena pelo eixo X."""
    points = []
    for row in rows:
        x = parse_number(row.get(x_column))
        y = parse_number(row.get(metric))
        if x is None or y is None:
            continue
        if log_scale and y <= 0:
            continue
        points.append((x / x_div, y))

    points.sort(key=lambda point: point[0])
    return points


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow}


def parse_expr(text) -> float:
    """Avalia expressões aritméticas simples, ex.: '5*1024**3' (sem usar eval)."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        raise ValueError(f"expressão inválida: {text!r}")
    return float(ev(ast.parse(str(text).strip(), mode="eval").body))


def resolve_x_exp(value, datasets, x_column: str) -> int:
    """Expoente k (base 2) do divisor do eixo X: inteiro dado ou 'auto' (múltiplo de 10)."""
    if str(value).lower() != "auto":
        return int(value)
    largest = 0.0
    for dataset in datasets:
        for rows in (dataset["train"], dataset["val"]):
            for row in rows:
                x = parse_number(row.get(x_column))
                if x is not None:
                    largest = max(largest, abs(x))
    k = 0
    while largest >= 2 ** (k + 10):
        k += 10
    return k


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


def focus_limits(values, finals_y, pct, log_scale):
    """Limites do eixo Y que contêm `pct`% centrais dos dados (e os valores finais).

    Retorna (lo, hi, n_fora) ou None se não houver o que ajustar.
    """
    if not 0 < pct < 100 or len(values) < 2:
        return None
    arr = np.asarray(values, dtype=float)
    tail = (100.0 - pct) / 2.0
    lo, hi = np.percentile(arr, [tail, 100.0 - tail])
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return None
    n_out = int(((arr < lo) | (arr > hi)).sum())
    lo_v = min([lo] + list(finals_y))
    hi_v = max([hi] + list(finals_y))
    if log_scale:
        pad = (hi_v / lo_v) ** 0.06
        return lo_v / pad, hi_v * pad, n_out
    pad = 0.06 * (hi_v - lo_v)
    return lo_v - pad, hi_v + pad, n_out


def place_labels(fig, ax, finals, marker_labels, final_fs=12, marker_fs=10):
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
            ha=ha, va=va, fontsize=final_fs if final else marker_fs,
            fontweight="bold" if final else "normal", color=color, zorder=5,
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7, "alpha": 0.8,
                        "shrinkA": 0, "shrinkB": 3} if (abs(dy) > marker_fs * 1.5 or final) else None,
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
        # Ponto fora da janela visível (eixo Y focado): não há o que rotular.
        if not (axbox.x0 - 1 <= anchor[0] <= axbox.x1 + 1 and
                axbox.y0 - 1 <= anchor[1] <= axbox.y1 + 1):
            return False

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

    # Deslocamentos proporcionais à fonte, para as caixas maiores não se sobreporem.
    fstep = final_fs * 1.9
    mstep = marker_fs * 1.2
    final_offsets = [(10, 0)] + [(10, s * fstep * k) for k in range(1, 9) for s in (1, -1)]
    marker_offsets = [(0, mstep), (0, -mstep), (0, 2.4 * mstep), (0, -2.4 * mstep),
                      (0, 3.8 * mstep), (0, -3.8 * mstep),
                      (10, mstep), (10, -mstep), (-10, mstep), (-10, -mstep)]

    for x, y, text, color in finals:
        place(x, y, text, color, final_offsets, final=True)
    for x, y, text, color in marker_labels:
        place(x, y, text, color, marker_offsets, final=False)


def _padded_extent_box(box, pad=3):
    return Bbox.from_extents(box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad)


def plot_metric(datasets, labels, metric: str, x_column: str, out_stem: Path,
                trend_only: bool, trend_window: int, markers: int = 8,
                label_markers: bool = False, y_ticks: int = 15,
                annotate_final: bool = True, focus_pct: float = 95.0,
                font_size: float = 13.0, x_exp: int = 10, formats=("png",),
                x_max=None, x_tick=None, show_title: bool = True,
                eval_trend_window: int = 1):
    # Eixo X com marcações em potência de 2 (ex.: 0..5 × 2^30): ticks inteiros e rótulo
    # compacto "[0, 1, ..., n] × 2^e", sem o nome da métrica.
    x_axis_label = None
    if x_max is not None and x_tick:
        exp_f, count_f = math.log2(x_tick), x_max / x_tick
        if abs(exp_f - round(exp_f)) < 1e-9 and abs(count_f - round(count_f)) < 1e-9 \
                and 1 <= round(count_f) <= 12:
            x_exp = int(round(exp_f))
            x_axis_label = "Tokens treinados (GiB)"
    x_div = float(2 ** x_exp)
    log_scale = metric in LOG_METRICS
    available = False
    fig, ax = plt.subplots(figsize=(12, 7))
    final_fs = font_size - 1
    marker_fs = font_size - 3
    all_values = []
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", [None])
    finals = []
    marker_labels = []
    marker_sets = []
    smoothed_any = False

    for index, dataset in enumerate(datasets):
        color = palette[index % len(palette)]
        label = labels[index]

        train_points = get_points(dataset["train"], x_column, metric, log_scale, x_div)
        val_points = get_points(dataset["val"], x_column, metric, log_scale, x_div)

        for split_name, points in (("Treino", train_points), ("Validação", val_points)):
            if not points:
                continue

            available = True
            is_train = split_name == "Treino"
            # Eval/validação: mesma cor do treino, porém mais clara.
            line_color = color if is_train else lighten(color)
            window = trend_window if is_train else eval_trend_window
            smooth = trend_only and metric not in NO_SMOOTH_METRICS
            shown_points = moving_average(points, window) if smooth else points
            xs, ys = zip(*shown_points)
            all_values.extend(ys)
            line_label = label if is_train else f"{label} (val)"
            legend_label = line_label
            # O ponto final fica sempre sobre a ponta da curva desenhada: valor bruto no
            # modo completo e média móvel no --trend-only (evita gap entre linha e ponto).
            final_x, final_y = xs[-1], ys[-1]
            if smooth and window > 1:
                smoothed_any = True

            if trend_only:
                ax.plot(xs, ys, label=legend_label, color=line_color, linewidth=2.2,
                        linestyle="-" if is_train else "--")
            else:
                ax.plot(xs, ys, label=legend_label, color=line_color,
                        alpha=0.85 if is_train else 0.95,
                        linewidth=1.5 if is_train else 1.8,
                        linestyle="-" if is_train else "--")

            # N marcadores ao longo da curva (opcionalmente com o valor escrito).
            idxs = marker_indices(len(xs), markers)
            if idxs:
                marker_sets.append(([xs[i] for i in idxs], [ys[i] for i in idxs],
                                    line_color, "o" if is_train else "s"))
                if label_markers:
                    for i in idxs[:-1]:  # o último recebe a anotação do valor final
                        marker_labels.append((xs[i], ys[i], format_value(ys[i]), line_color))

            # Valor final em destaque (o texto é posicionado depois, sem colisão).
            if annotate_final:
                ax.plot([final_x], [final_y], marker="o" if is_train else "s",
                        markersize=8, color=line_color, markeredgecolor="black",
                        markeredgewidth=0.8, zorder=4, clip_on=False)
                finals.append((final_x, final_y, format_value(final_y), line_color))

    if not available:
        plt.close(fig)
        return False

    label = metric_label(metric)
    x_name = metric_label(x_column) if x_column in LABELS else x_column.replace("_", " ")
    x_label = x_axis_label or (x_name + (f" (× $2^{{{x_exp}}}$)" if x_exp else ""))
    ax.set_xlabel(x_label, fontsize=font_size + 1)
    ax.set_ylabel(label, fontsize=font_size + 1)
    title = f"{label} por {x_name}" + (" — tendência" if trend_only else "")

    if log_scale:
        ax.set_yscale("log")
    elif y_ticks > 0:
        # Mais marcações no eixo Y, com rótulos de precisão adequada.
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: format_value(v)))

    if x_max is not None:
        # Eixo X fixo de 0 até x_max, com marcações a cada x_tick (valores brutos).
        ax.set_xlim(0, x_max / x_div)
        if x_tick:
            ticks = np.arange(0, x_max + x_tick * 0.5, x_tick) / x_div
            ax.xaxis.set_major_locator(FixedLocator(ticks))
    elif annotate_final:
        ax.margins(x=0.12)  # espaço à direita para as anotações finais

    # Foco no intervalo de interesse: eixo Y limitado aos `focus_pct`% centrais dos dados.
    focus = focus_limits(all_values, [f[1] for f in finals], focus_pct, log_scale)
    if focus:
        lo, hi, n_out = focus
        ax.set_ylim(lo, hi)
        title += (f"\nEixo Y: {focus_pct:g}% centrais dos dados "
                  f"[{format_value(lo)} – {format_value(hi)}]"
                  + (f" · {n_out} pontos fora" if n_out else ""))
    if trend_only and smoothed_any and annotate_final:
        title += (f"\nValores finais: média móvel de {trend_window} pontos (treino)"
                  f" / {eval_trend_window} (eval)")
    if show_title:
        ax.set_title(title, fontsize=font_size + 2)

    # Marcadores ao longo das curvas: só os que estão dentro da janela visível.
    (x_lo, x_hi), (y_lo, y_hi) = ax.get_xlim(), ax.get_ylim()
    for mx, my, mcolor, mshape in marker_sets:
        keep = [(a, b) for a, b in zip(mx, my)
                if x_lo <= a <= x_hi and y_lo <= b <= y_hi]
        if keep:
            ax.plot(*zip(*keep), linestyle="none", marker=mshape, markersize=4.5,
                    color=mcolor, zorder=3, clip_on=False)

    ax.tick_params(axis="both", which="major", labelsize=font_size - 1)
    # Eixo X em números simples (já divididos por 2^k), sem multiplicador "1e9" no canto.
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{round(v)}" if abs(v - round(v)) < 1e-6 else f"{v:.4g}"))
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=font_size - 2)
    fig.tight_layout()
    place_labels(fig, ax, finals, marker_labels, final_fs, marker_fs)
    for fmt in formats:
        out_path = out_stem.with_name(f"{out_stem.name}.{fmt}")
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"Salvo: {out_path}")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Gera gráficos comparativos a partir de múltiplos CSVs do H-Net."
    )
    parser.add_argument(
        "--csv", nargs="+", required=True, type=Path, metavar="ARQUIVO.csv",
        help="Um ou mais CSVs para comparar (treino e/ou eval do eval_pt.py).",
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
        help="Janela da média móvel, em pontos, para curvas de treino "
             "(padrão: 15; use 1 para sem suavização).",
    )
    parser.add_argument(
        "--eval-trend-window", type=int, default=1,
        help="Janela da média móvel, em pontos, para curvas de eval/validação "
             "(padrão: 1 = sem suavização).",
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
        "--y-ticks", type=int, default=6, metavar="N",
        help="Número aproximado de marcações no eixo Y (padrão: 6; 0 mantém o automático).",
    )
    parser.add_argument(
        "--no-final-annotation", action="store_true",
        help="Não anota o valor final de cada curva.",
    )
    parser.add_argument(
        "--focus-pct", type=float, default=95.0, metavar="P",
        help="Ajusta o eixo Y ao intervalo central que contém P%% dos dados "
             "(padrão: 95; 0 ou 100 desliga e usa o eixo completo).",
    )
    parser.add_argument(
        "--font-size", type=float, default=13.0, metavar="PT",
        help="Tamanho-base da fonte em pontos (padrão: 13). Título, eixos, ticks, "
             "legenda e rótulos escalam a partir dele.",
    )
    parser.add_argument(
        "--format", nargs="+", default=["png"], choices=["png", "pdf", "svg"],
        dest="formats", metavar="FMT",
        help="Formato(s) de saída: png, pdf, svg (padrão: png; ex.: --format png pdf).",
    )
    parser.add_argument(
        "--x-exp", default="10", metavar="K",
        help="Divide o eixo X por 2^K e indica '× 2^K' no rótulo (padrão: 10, ou seja "
             "N × 2^10; 0 = sem escala; 'auto' escolhe 10, 20, 30... pelo maior valor).",
    )
    parser.add_argument(
        "--no-title", action="store_true",
        help="Remove o título do gráfico (útil quando a legenda vem do documento).",
    )
    parser.add_argument(
        "--x-max", default="default", metavar="EXPR",
        help="Limite superior do eixo X em valores brutos (expressão aceita, ex.: 5*1024**3). "
             "Padrão: 5*1024**3 quando o eixo X é 'tokens'; 'auto' deixa automático.",
    )
    parser.add_argument(
        "--x-tick", default="default", metavar="EXPR",
        help="Espaçamento das marcações do eixo X em valores brutos (ex.: 1024**3). "
             "Padrão: 1024**3 quando o eixo X é 'tokens'; 'auto' deixa automático.",
    )
    args = parser.parse_args()

    if str(args.x_exp).lower() != "auto":
        try:
            if int(args.x_exp) < 0:
                raise ValueError
        except ValueError:
            parser.error("--x-exp deve ser um inteiro >= 0 ou 'auto'.")
    if not 0 <= args.focus_pct <= 100:
        parser.error("--focus-pct deve estar entre 0 e 100.")
    if args.font_size < 4:
        parser.error("--font-size deve ser pelo menos 4.")
    if args.trend_window < 1 or args.eval_trend_window < 1:
        parser.error("--trend-window e --eval-trend-window devem ser maiores ou iguais a 1.")
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
    x_exp = resolve_x_exp(args.x_exp, datasets, args.x_column)
    defaults = {"x_max": 5 * 1024 ** 3, "x_tick": 1024 ** 3} if args.x_column == "tokens" else {}
    axis_limits = {}
    for name, raw in (("x_max", args.x_max), ("x_tick", args.x_tick)):
        if str(raw).lower() == "auto":
            axis_limits[name] = None
        elif str(raw).lower() == "default":
            axis_limits[name] = defaults.get(name)
        else:
            try:
                axis_limits[name] = parse_expr(raw)
            except (ValueError, SyntaxError):
                parser.error(f"--{name.replace('_', '-')} inválido: {raw!r}")
            if axis_limits[name] <= 0:
                parser.error(f"--{name.replace('_', '-')} deve ser positivo.")
    plt.rcParams["pdf.fonttype"] = 42  # fontes embutidas como TrueType (texto selecionável)
    plt.rcParams["svg.fonttype"] = "none"

    for metric in metrics:
        out_stem = args.out_dir / safe_filename(metric)
        if plot_metric(
            datasets, labels, metric, args.x_column, out_stem,
            trend_only=args.trend_only, trend_window=args.trend_window,
            eval_trend_window=args.eval_trend_window,
            markers=args.markers, label_markers=args.label_markers,
            y_ticks=args.y_ticks, annotate_final=not args.no_final_annotation,
            focus_pct=args.focus_pct, font_size=args.font_size,
            x_exp=x_exp, formats=args.formats,
            x_max=axis_limits["x_max"], x_tick=axis_limits["x_tick"],
            show_title=not args.no_title,
        ):
            generated += 1

    if generated == 0:
        print("Nenhum gráfico foi gerado: não há métricas numéricas válidas em comum com os dados carregados.")
    else:
        mode = "tendências" if args.trend_only else "séries completas"
        print(f"\nConcluído: {generated} gráfico(s) ({mode}) em {args.out_dir}")


if __name__ == "__main__":
    main()