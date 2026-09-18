from pathlib import Path
from typing import Iterable
import argparse
import math
import re

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist


# ============================================================
# GRID FIXO
# ============================================================
#
# O texto é quebrado em um grid fixo (em vez de uma tira horizontal
# única, que ficava ilegível e exigia figuras cada vez mais largas):
#
#   GRID_COLS = 40  → colunas por linha
#   GRID_ROWS = 12  → linhas máximas
#
# posição i do texto → linha i // GRID_COLS, coluna i % GRID_COLS
# ============================================================

GRID_COLS = 40
GRID_ROWS = 12
MAX_POSITIONS = GRID_ROWS * GRID_COLS


# ============================================================
# LEITURA DOS ARQUIVOS
# ============================================================

def read_step_file(path: Path):
    """
    Lê arquivos no formato:

        # [stage 0]
        0 | 0x53 | S | 0.12345 | 0
        1 | 0x65 | e | 0.23456 | 0

        # [stage 1]
        0 | 0.91234 | 1
        ...

    (também aceita o formato antigo "Stage 0" sem colchetes/#)

    Retorna:

        {
            0: {"pos": [...], "byte": [...], "char": [...], "prob": [...], "ind": [...]},
            1: {"pos": [...], "byte": [None,...], "char": [None,...], "prob": [...], "ind": [...]},
            ...
        }
    """

    stages = {}
    current_stage = None

    with open(path, "r", encoding="utf-8") as f:

        for line in f:
            line = line.rstrip("\n")

            if not line.strip():
                continue

            # ------------------------------------------------
            # Detecta "Stage 0", "# [stage 0]", "[Stage 1]", ...
            # ------------------------------------------------

            match = re.match(
                r"^\s*#*\s*\[?\s*stage\s+(\d+)\s*\]?",
                line,
                re.IGNORECASE,
            )

            if match:
                current_stage = int(match.group(1))

                stages[current_stage] = {
                    "pos": [],
                    "byte": [],   # None para estágios sem byte/char (stage >= 1)
                    "char": [],   # None para estágios sem byte/char (stage >= 1)
                    "prob": [],
                    "ind": [],
                }

                continue

            if current_stage is None:
                continue

            # ------------------------------------------------
            # Dados. Dois formatos possíveis:
            #
            #   stage 0 (byte-level):        pos | byte | char | prob | ind
            #   stage >= 1 (chunk-level):    chunk_pos | prob | ind
            # ------------------------------------------------

            parts = [p.strip() for p in line.split("|")]

            if len(parts) == 5:
                try:
                    pos = int(parts[0])
                    byte_val = int(parts[1], 16)
                    char_repr = parts[2] if parts[2] != "" else " "
                    prob = float(parts[3])
                    ind = int(parts[4])
                except ValueError:
                    continue

            elif len(parts) == 3:
                try:
                    pos = int(parts[0])
                    byte_val = None
                    char_repr = None
                    prob = float(parts[1])
                    ind = int(parts[2])
                except ValueError:
                    continue

            else:
                continue

            stages[current_stage]["pos"].append(pos)
            stages[current_stage]["byte"].append(byte_val)
            stages[current_stage]["char"].append(char_repr)
            stages[current_stage]["prob"].append(prob)
            stages[current_stage]["ind"].append(ind)

    return stages


def find_step_files(input_dir: Path):
    """Encontra step_100.txt, step_200.txt, ... e ordena numericamente."""

    files = []

    for path in input_dir.glob("step_*.txt"):
        match = re.match(r"^step_(\d+)\.txt$", path.name)
        if match:
            step = int(match.group(1))
            files.append((step, path))

    files.sort(key=lambda x: x[0])
    return files


def rows_needed(n_positions: int, grid_cols: int = GRID_COLS) -> int:
    """Quantas linhas do grid são necessárias para n_positions itens."""
    if n_positions <= 0:
        return 1
    return min(math.ceil(n_positions / grid_cols), GRID_ROWS)


# ============================================================
# CONECTORES ("U" entre posições consecutivas)
# ============================================================

def draw_point_connectors(
    ax,
    positions: list,
    grid_cols: int,
    row_height: float,
    y_line_offset: float,
    y_tick_offset: float,
    color: str,
    connector_gap: float,
):
    """
    Conecta posições GLOBAIS (0-based) consecutivas de `positions` com um
    "U": desce do ponto anterior, atravessa, sobe no ponto seguinte — com
    um pequeno vão (connector_gap) de cada lado, para o fim de um U não
    encostar no início do próximo.

    Um par que atravessa a quebra de linha do grid vira duas setas (uma
    saindo até a borda direita da primeira linha, outra entrando pela
    esquerda da linha seguinte).
    """
    for start_pos, end_pos in zip(positions[:-1], positions[1:]):
        row_start = start_pos // grid_cols
        row_end = end_pos // grid_cols

        if row_start == row_end:
            row = row_start
            y_top = -row * row_height
            y_line = y_top + y_line_offset
            y_line_top = y_top + y_tick_offset

            x_start = float(start_pos % grid_cols) + connector_gap
            x_end = float(end_pos % grid_cols) - connector_gap

            if x_end > x_start:
                ax.plot([x_start, x_end], [y_line, y_line], color=color, linewidth=1.5)
                ax.plot([x_start, x_start], [y_line, y_line_top], color=color, linewidth=1.5)
                ax.plot([x_end, x_end], [y_line, y_line_top], color=color, linewidth=1.5)
            else:
                # pontos adjacentes demais pro gap — só um tick, sem barra
                ax.plot(
                    [float(start_pos % grid_cols)] * 2,
                    [y_line, y_line_top],
                    color=color, linewidth=1.5,
                )
            continue

        # Atravessa quebra de linha: seta de saída na 1ª linha, seta de
        # entrada na linha seguinte. Não desenha nada nas linhas do meio
        # (não há ponto real ali, só passagem).
        for row in (row_start, row_end):
            y_top = -row * row_height
            y_line = y_top + y_line_offset
            y_line_top = y_top + y_tick_offset

            if row == row_start:
                x_start = float(start_pos % grid_cols) + connector_gap
                x_edge = float(grid_cols - 1) - connector_gap
                if x_edge > x_start:
                    ax.plot([x_start, x_start], [y_line, y_line_top], color=color, linewidth=1.5)
                    ax.annotate(
                        "",
                        xy=(min(x_edge + 0.6, grid_cols - 0.2), y_line),
                        xytext=(x_start, y_line),
                        arrowprops=dict(arrowstyle="->", color=color, linewidth=1.5),
                    )
            else:
                x_end = float(end_pos % grid_cols) - connector_gap
                x_edge = max(-0.8, x_end - 1.0)
                if x_end > x_edge:
                    ax.plot([x_end, x_end], [y_line, y_line_top], color=color, linewidth=1.5)
                    ax.annotate(
                        "",
                        xy=(x_end, y_line),
                        xytext=(x_edge, y_line),
                        arrowprops=dict(arrowstyle="->", color=color, linewidth=1.5),
                    )


# ============================================================
# DESENHO
# ============================================================

# Offsets verticais (relativos ao topo de cada linha física do grid)
OFF_BYTE = 0.92
OFF_CHAR = 0.48
OFF_BOX = 0.0
OFF_LINE = -0.42          # linha azul (boundaries do stage 0)
OFF_LINE_TOP = -0.24       # tick da linha azul, perto da base do box
OFF_CHILD_BOX = -0.68      # box pequeno do stage 1, abaixo da linha azul
OFF_CHILD_LINE = -1.02     # linha vermelha (boundaries do stage 1)
OFF_CHILD_LINE_TOP = -0.86  # tick da linha vermelha, perto da base do box pequeno

ROW_HEIGHT_NO_CHILD = 1.7
ROW_HEIGHT_WITH_CHILD = 2.3

BOX_MARKER_SIZE = 190
CHILD_MARKER_SIZE = 90
CONNECTOR_GAP = 0.16


def draw_stage(
    ax,
    data: dict,
    child_ind: list | None = None,
    grid_cols: int = GRID_COLS,
    grid_rows: int = GRID_ROWS,
    title: str = "",
):
    """
    Desenha, num único painel:

        44   75   72 ...            ← byte hexadecimal
         D    u    r ...             ← caractere
        [■]  [ ]  [ ]...             ← box do stage 0 (verde = boundary)
         └────┘    └──►              ← "U" azul entre boundaries do stage 0
        [ ]       [■]                ← box pequeno do stage 1, um por
                                        boundary do stage 0 (alinhado na
                                        mesma coluna), verde se também for
                                        boundary do stage 1
                    └──────┘         ← "U" vermelho entre boundaries do
                                        stage 1 (conecta só os boxes
                                        pequenos verdes)

    `data` é o dict do stage 0 (com byte/char reais). `child_ind` é o
    boundary_ind do stage 1 — se None, os boxes/linha vermelha não são
    desenhados (comportamento de um modelo de 1 estágio só).
    """

    chars = data["char"]
    bytes_ = data["byte"]
    inds = list(data["ind"])  # cópia local — não altera os dados originais

    n = len(chars)

    if n == 0:
        ax.axis("off")
        return 1

    # O primeiro índice é sempre um boundary visualmente (não há contexto
    # anterior). Isso é só cosmético — os cálculos de chunk usam o ind
    # original (raw_ind), não esta cópia.
    raw_ind = inds.copy()
    inds[0] = 1

    truncated = n > grid_rows * grid_cols
    n_draw = min(n, grid_rows * grid_cols)

    has_child = child_ind is not None
    row_height = ROW_HEIGHT_WITH_CHILD if has_child else ROW_HEIGHT_NO_CHILD
    n_rows_used = rows_needed(n_draw, grid_cols)

    # --------------------------------------------------------
    # Posições reais de boundary do stage 0 (SEM o índice 0 forçado —
    # essas posições são as que de fato viram "chunks" alimentados no
    # stage 1, então têm que bater 1:1 com child_ind).
    # --------------------------------------------------------

    boundary_positions = [i for i in range(n_draw) if raw_ind[i] >= 0.5]

    if has_child and len(boundary_positions) != len(child_ind):
        print(
            f"[draw_stage] Aviso: stage 0 tem {len(boundary_positions)} "
            f"boundaries mas o próximo estágio tem {len(child_ind)} "
            f"entradas — deveriam ser iguais. Linha vermelha não será "
            f"desenhada neste frame."
        )
        has_child = False

    # --------------------------------------------------------
    # Byte / char / box do stage 0
    # --------------------------------------------------------

    for i in range(n_draw):
        row = i // grid_cols
        col = i % grid_cols
        x = float(col)
        y_top = -row * row_height

        is_boundary = inds[i] >= 0.5

        ax.scatter(
            x, y_top + OFF_BOX,
            s=BOX_MARKER_SIZE,
            marker="s",
            facecolor="green" if is_boundary else "none",
            edgecolor="green" if is_boundary else "black",
            linewidths=1.0 if is_boundary else 0.8,
            zorder=3,
        )

        char = chars[i]
        if char == "" or char is None:
            char = " "
        ax.text(x, y_top + OFF_CHAR, char, ha="center", va="center", fontsize=11)

        ax.text(
            x, y_top + OFF_BYTE,
            f"{bytes_[i]:02X}",
            ha="center", va="center", fontsize=7,
        )

    # --------------------------------------------------------
    # "U" azul: conecta boundaries consecutivos do stage 0
    # (usa inds com índice 0 forçado — é o comportamento visual já
    # validado antes, mantido para não regredir o stage 0 sozinho)
    # --------------------------------------------------------

    display_boundary_positions = [i for i in range(n_draw) if inds[i] >= 0.5]
    if len(display_boundary_positions) >= 2:
        draw_point_connectors(
            ax, display_boundary_positions, grid_cols, row_height,
            OFF_LINE, OFF_LINE_TOP, color="blue", connector_gap=CONNECTOR_GAP,
        )

    # --------------------------------------------------------
    # Boxes pequenos do stage 1 — um por boundary REAL do stage 0,
    # exatamente na mesma coluna. Verde se também for boundary do
    # stage 1 (child_ind[i] == 1).
    # --------------------------------------------------------

    if has_child:
        child_boundary_positions = []

        for idx, pos in enumerate(boundary_positions):
            row = pos // grid_cols
            col = pos % grid_cols
            x = float(col)
            y_top = -row * row_height

            is_child_boundary = child_ind[idx] >= 0.5
            if is_child_boundary:
                child_boundary_positions.append(pos)

            ax.scatter(
                x, y_top + OFF_CHILD_BOX,
                s=CHILD_MARKER_SIZE,
                marker="s",
                facecolor="green" if is_child_boundary else "none",
                edgecolor="green" if is_child_boundary else "black",
                linewidths=0.9 if is_child_boundary else 0.7,
                zorder=3,
            )

        # --------------------------------------------------------
        # "U" vermelho: conecta só os boxes pequenos que são boundary
        # do stage 1 — ponto a ponto, igual ao azul, sem cobrir o
        # intervalo inteiro (evita o bug de um U não bater no outro).
        # --------------------------------------------------------

        if len(child_boundary_positions) >= 2:
            draw_point_connectors(
                ax, child_boundary_positions, grid_cols, row_height,
                OFF_CHILD_LINE, OFF_CHILD_LINE_TOP,
                color="red", connector_gap=CONNECTOR_GAP,
            )

    # --------------------------------------------------------
    # Título e eixo
    # --------------------------------------------------------

    full_title = title
    if truncated:
        full_title += f"  (mostrando {n_draw} de {n} — truncado em {MAX_POSITIONS})"

    if full_title:
        ax.set_title(full_title, loc="left", fontsize=12, fontweight="bold", pad=8)

    bottom_pad = abs(OFF_CHILD_LINE) + 0.35 if has_child else abs(OFF_LINE) + 0.35

    ax.set_xlim(-1, grid_cols + 1)
    ax.set_ylim(-(n_rows_used - 1) * row_height - bottom_pad, 1.15)
    ax.axis("off")

    return n_rows_used, row_height


# ============================================================
# GIF
# ============================================================

def create_gif(
    model: str,
    phrase: int,
    output_dir: Path | None = None,
    output_file: Path | None = None,
    fps: int = 5,
):
    """
    Cria text/output/<model>/frase_<phrase>.gif a partir de
    text/output/<model>/<phrase>/step_N.txt

    Painel único: stage 0 (byte-level) com o stage 1 (se existir)
    embutido como uma segunda camada de boxes/conectores, alinhada às
    mesmas colunas dos boundaries do stage 0.
    """

    if output_dir is None:
        output_dir = Path("text") / "output" / model / str(phrase)
    else:
        output_dir = Path(output_dir)

    step_files = find_step_files(output_dir)

    if not step_files:
        raise RuntimeError(f"Nenhum arquivo step_N.txt encontrado em:\n{output_dir}")

    print(f"Encontrados {len(step_files)} steps")

    frames = []
    for step, path in step_files:
        stages = read_step_file(path)
        if 0 not in stages:
            print(f"Aviso: nenhum Stage 0 encontrado em {path}")
            continue
        frames.append({"step": step, "stages": stages})

    if not frames:
        raise RuntimeError("Nenhum frame válido foi encontrado.")

    has_child_any = any(1 in frame["stages"] for frame in frames)
    row_height = ROW_HEIGHT_WITH_CHILD if has_child_any else ROW_HEIGHT_NO_CHILD

    # --------------------------------------------------------
    # Linhas de grid necessárias (baseado no stage 0 — o texto é o
    # mesmo em todos os passos, então isso é estável entre frames)
    # --------------------------------------------------------

    max_rows_used = 1
    any_truncated = False
    for frame in frames:
        n = len(frame["stages"][0]["char"])
        if n > MAX_POSITIONS:
            any_truncated = True
        max_rows_used = max(max_rows_used, rows_needed(n))

    if any_truncated:
        print(
            f"Aviso: o texto excede {MAX_POSITIONS} posições "
            f"(grid {GRID_ROWS}x{GRID_COLS}) — será truncado nos frames."
        )

    # --------------------------------------------------------
    # Figura — painel único
    # --------------------------------------------------------

    fig_width = 16.0
    row_height_in = 0.42
    header_in = 0.7

    fig_height = header_in + max_rows_used * row_height_in * (
        row_height / ROW_HEIGHT_NO_CHILD
    )

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    # ========================================================
    # UPDATE
    # ========================================================

    def update(frame_index: int) -> Iterable[Artist]:
        frame = frames[frame_index]
        ax.clear()

        child_ind = frame["stages"][1]["ind"] if 1 in frame["stages"] else None
        title = "Stage 0"
        if child_ind is not None:
            title += "   [vermelho: boundaries do Stage 1]"

        draw_stage(
            ax=ax,
            data=frame["stages"][0],
            child_ind=child_ind,
            grid_cols=GRID_COLS,
            grid_rows=GRID_ROWS,
            title=title,
        )

        fig.suptitle(
            f"{model} — frase {phrase} — Step {frame['step']}",
            fontsize=14,
            fontweight="bold",
        )

        fig.subplots_adjust(left=0.03, right=0.98, top=0.85, bottom=0.05)

        return tuple(ax.get_children())

    # ========================================================
    # ANIMATION
    # ========================================================

    animation = FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000 / fps, repeat=True, blit=False,
    )

    if output_file is None:
        output_file = Path("text") / "output" / model / f"frase_{phrase}.gif"
    else:
        output_file = Path(output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("Salvando GIF em:")
    print(output_file)

    animation.save(output_file, writer=PillowWriter(fps=fps))

    plt.close(fig)

    print("GIF criado com sucesso.")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gera GIF da evolução dos boundaries do H-Net."
    )
    parser.add_argument("model", help="Nome do modelo")
    parser.add_argument("phrase", type=int, choices=range(1, 5), help="Número da frase (1-4)")
    parser.add_argument("--fps", type=int, default=5, help="Frames por segundo")
    parser.add_argument("--input-dir", default=None, help="Diretório contendo os step_N.txt")
    parser.add_argument("--output", default=None, help="Arquivo GIF de saída")

    args = parser.parse_args()

    create_gif(
        model=args.model,
        phrase=args.phrase,
        output_dir=args.input_dir,
        output_file=args.output,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()