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
# Em vez de desenhar toda a sequência numa única tira horizontal
# (que ficava ilegível e exigia figuras cada vez mais largas para
# textos longos), o texto é quebrado em um grid fixo:
#
#   GRID_COLS = 40  → colunas por linha
#   GRID_ROWS = 12  → linhas máximas por estágio
#
# posição i do texto → linha i // GRID_COLS, coluna i % GRID_COLS
#
# Isso mantém o tamanho do frame estável entre passos (importante
# para o GIF não "pular" de escala) e legível para textos de até
# GRID_ROWS * GRID_COLS = 480 bytes por estágio.
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

        Stage 0
        0 | 0x53 | S | 0.12345 | 0
        1 | 0x65 | e | 0.23456 | 0
        2 | 0x6C | l | 0.91234 | 1

        Stage 1
        ...

    Retorna:

        {
            0: {
                "pos": [...],
                "byte": [...],
                "char": [...],
                "prob": [...],
                "ind": [...]
            },
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
            #   stage 0 (byte-level):  pos | byte | char | prob | ind
            #   stage >= 1 (chunk-level, sem byte/char):  chunk_pos | prob | ind
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
    """
    Encontra:

        step_100.txt
        step_200.txt
        step_300.txt

    e ordena numericamente.
    """

    files = []

    for path in input_dir.glob("step_*.txt"):

        match = re.match(
            r"^step_(\d+)\.txt$",
            path.name,
        )

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


def compute_chunk_spans(ind: list) -> list:
    """
    Reconstrói os chunks de um estágio a partir do seu boundary_ind, onde
    o boundary marca o FIM de cada chunk (mesma convenção do resto do
    código).

    Ex.: ind = [0,0,1,0,1]  →  spans = [(0,2), (3,4)]

    Retorna lista de (start, end) inclusivos, em unidades de posição
    NESTE estágio (para o stage 0, são colunas de byte; para estágios
    mais internos, são "chunk_pos" do próprio estágio).
    """
    spans = []
    start = 0
    for i, b in enumerate(ind):
        if b >= 0.5:
            spans.append((start, i))
            start = i + 1
    # chunk final sem boundary explícito (não deveria acontecer se o
    # último elemento sempre fecha um chunk, mas protege contra dados
    # incompletos sem quebrar o desenho)
    if start < len(ind):
        spans.append((start, len(ind) - 1))
    return spans


def map_boundaries_to_parent_columns(chunk_spans: list, child_ind: list):
    """
    Agrupa os spans do estágio PAI (ex.: spans de byte do stage 0) em
    super-chunks, usando o boundary_ind do estágio FILHO (ex.: stage 1),
    cujas posições (chunk_pos) referenciam 1:1 os chunks do estágio pai.

    Ou seja: cada posição i de `child_ind` corresponde ao i-ésimo chunk de
    `chunk_spans`. Quando child_ind[i] == 1, fecha-se um super-chunk que
    agrupa todos os spans desde o último boundary do filho até aqui —
    o range resultante é (início do primeiro span do grupo, fim do
    último span do grupo), já na escala de colunas do estágio pai.

    Levanta ValueError se as contagens não baterem (sinal de arquivo
    inconsistente/corrompido, não de bug de lógica).
    """
    if len(chunk_spans) != len(child_ind):
        raise ValueError(
            f"Descompasso entre estágios: pai tem {len(chunk_spans)} chunks, "
            f"filho tem {len(child_ind)} entradas — deveriam ser iguais "
            f"(cada posição do filho referencia um chunk do pai)."
        )

    parent_ranges = []
    group_start_idx = 0
    for i, b in enumerate(child_ind):
        if b >= 0.5:
            start_col = chunk_spans[group_start_idx][0]
            end_col = chunk_spans[i][1]
            parent_ranges.append((start_col, end_col))
            group_start_idx = i + 1

    if group_start_idx < len(chunk_spans):
        start_col = chunk_spans[group_start_idx][0]
        end_col = chunk_spans[-1][1]
        parent_ranges.append((start_col, end_col))

    return parent_ranges


# ============================================================
# DESENHO
# ============================================================

def draw_connector_layer(
    ax,
    ranges: list,
    grid_cols: int,
    row_height: float,
    y_line_offset: float,
    y_tick_offset: float,
    color: str,
    connector_gap: float,
):
    """
    Desenha conectores em forma de "U" para uma lista de (start_col, end_col)
    em colunas GLOBAIS (0-based, antes da quebra em linhas do grid).

    Um range que atravessa a quebra de linha do grid é dividido: uma seta
    sai até a borda direita da primeira linha, outra entra pela borda
    esquerda da linha seguinte; linhas inteiramente cobertas pelo range
    (range muito longo, atravessando 2+ linhas) recebem uma reta simples.
    """
    for start_col, end_col in ranges:
        row_start = start_col // grid_cols
        row_end = end_col // grid_cols

        if row_start == row_end:
            row = row_start
            y_top = -row * row_height
            y_line = y_top + y_line_offset
            y_line_top = y_top + y_tick_offset

            x_start = float(start_col % grid_cols) + connector_gap
            x_end = float(end_col % grid_cols) - connector_gap

            if x_end > x_start:
                ax.plot([x_start, x_end], [y_line, y_line], color=color, linewidth=1.5)
                ax.plot([x_start, x_start], [y_line, y_line_top], color=color, linewidth=1.5)
                ax.plot([x_end, x_end], [y_line, y_line_top], color=color, linewidth=1.5)
            else:
                # chunk de largura 0/1 — sem espaço pro gap, desenha só o tick
                ax.plot(
                    [float(start_col % grid_cols)] * 2,
                    [y_line, y_line_top],
                    color=color, linewidth=1.5,
                )
            continue

        # Range atravessa uma ou mais quebras de linha do grid
        for row in range(row_start, row_end + 1):
            y_top = -row * row_height
            y_line = y_top + y_line_offset
            y_line_top = y_top + y_tick_offset

            if row == row_start:
                x_start = float(start_col % grid_cols) + connector_gap
                x_edge = float(grid_cols - 1) - connector_gap
                if x_edge > x_start:
                    ax.plot([x_start, x_start], [y_line, y_line_top], color=color, linewidth=1.5)
                    ax.annotate(
                        "",
                        xy=(min(x_edge + 0.6, grid_cols - 0.2), y_line),
                        xytext=(x_start, y_line),
                        arrowprops=dict(arrowstyle="->", color=color, linewidth=1.5),
                    )
            elif row == row_end:
                x_end = float(end_col % grid_cols) - connector_gap
                x_edge = max(-0.8, x_end - 1.0)
                if x_end > x_edge:
                    ax.plot([x_end, x_end], [y_line, y_line_top], color=color, linewidth=1.5)
                    ax.annotate(
                        "",
                        xy=(x_end, y_line),
                        xytext=(x_edge, y_line),
                        arrowprops=dict(arrowstyle="->", color=color, linewidth=1.5),
                    )
            else:
                # linha inteiramente coberta pelo range
                x_start = 0.0 + connector_gap
                x_end = float(grid_cols - 1) - connector_gap
                if x_end > x_start:
                    ax.plot([x_start, x_end], [y_line, y_line], color=color, linewidth=1.5)


def draw_stage(
    ax,
    stage: int,
    data: dict,
    grid_cols: int = GRID_COLS,
    grid_rows: int = GRID_ROWS,
    child_ind: list | None = None,
    child_stage_label: int | None = None,
):
    """
    Desenha um Stage em um grid de grid_rows x grid_cols, no estilo:

        53  65  6C  65  63          ← byte hexadecimal
         S   e   l   e   c          ← caractere
        [■] [ ] [ ] [■] [ ]         ← boundary (verde=1, vazio=0)
         └───┘         └──►         ← linha conectando boundaries vizinhos (azul)
         └───────┘                  ← linha do PRÓXIMO estágio, mapeada de
                                       volta pra estas colunas (vermelho),
                                       só desenhada se child_ind for passado

        ... (quebra a cada grid_cols posições, até grid_rows linhas) ...

    Cada linha física do grid corresponde a um bloco de até grid_cols
    posições consecutivas do texto. As linhas de chunk (conectores azuis)
    são desenhadas dentro de cada linha física; um chunk que atravessaria
    a quebra de linha não é conectado visualmente entre linhas.
    """

    chars = data["char"]
    bytes_ = data["byte"]
    probs = data["prob"]
    inds = list(data["ind"])  # cópia local — não altera os dados originais

    n = len(chars)

    if n == 0:
        ax.axis("off")
        return

    # Stage 0 tem byte/char reais; estágios mais internos (chunk-level)
    # têm byte=None/char=None (formato de 3 colunas no arquivo) — nesses
    # casos mostramos o índice do chunk e a probabilidade no lugar.
    is_byte_stage = any(b is not None for b in bytes_)

    # O primeiro índice é sempre um boundary (não há contexto anterior),
    # independentemente do que o arquivo registrou.
    inds[0] = 1

    truncated = n > grid_rows * grid_cols
    n_draw = min(n, grid_rows * grid_cols)

    # --------------------------------------------------------
    # Configuração visual
    # --------------------------------------------------------

    box_size = 0.28  # mantido só para referência de espaçamento; tamanho visual real é marker_size (pontos)
    marker_size = 190  # área do marcador em pontos^2 — ajuste aqui se quiser caixas maiores/menores
    row_height = 1.7  # espaço vertical reservado por linha física do grid
    connector_gap = 0.16  # vão (em colunas) entre o fim de um "U" e o início do próximo

    # Offsets verticais relativos ao topo de cada linha física
    off_box = 0.0
    off_char = 0.48
    off_byte = 0.92
    off_line = -0.42
    off_line_top = -0.24

    n_rows_used = rows_needed(n_draw, grid_cols)

    # --------------------------------------------------------
    # Desenha cada posição
    # --------------------------------------------------------

    for i in range(n_draw):

        row = i // grid_cols
        col = i % grid_cols

        x = float(col)
        y_top = -row * row_height  # linha 0 no topo, linhas seguintes descem

        y_box = y_top + off_box
        y_char = y_top + off_char
        y_byte = y_top + off_byte

        is_boundary = inds[i] >= 0.5

        # ----------------------------------------------------
        # Caixa (boundary = verde preenchido / não-boundary = preto vazio)
        # scatter com marker="s": tamanho em pontos (tela), sempre um
        # quadrado real, independente da escala/aspect ratio dos eixos.
        # ----------------------------------------------------

        if is_boundary:
            ax.scatter(
                x, y_box,
                s=marker_size,
                marker="s",
                facecolor="green",
                edgecolor="green",
                linewidths=1.0,
                zorder=3,
            )
        else:
            ax.scatter(
                x, y_box,
                s=marker_size,
                marker="s",
                facecolor="none",
                edgecolor="black",
                linewidths=0.8,
                zorder=3,
            )

        # ----------------------------------------------------
        # Linha de texto de cima: caractere (stage 0) ou índice
        # do chunk (estágios internos, sem char real)
        # ----------------------------------------------------

        if is_byte_stage:
            char = chars[i]
            if char == "" or char is None:
                char = " "
        else:
            char = f"{probs[i]:.2f}"

        ax.text(
            x,
            y_char,
            char,
            ha="center",
            va="center",
            fontsize=11 if is_byte_stage else 8,
        )

        # ----------------------------------------------------
        # Linha de texto de baixo: byte hexadecimal (stage 0) ou
        # índice do chunk (estágios internos, sem byte real)
        # ----------------------------------------------------

        if is_byte_stage:
            top_label = f"{bytes_[i]:02X}"
        else:
            top_label = f"#{i}"

        ax.text(
            x,
            y_byte,
            top_label,
            ha="center",
            va="center",
            fontsize=7,
        )

    # ========================================================
    # LIGAÇÕES ENTRE BOUNDARIES (uma vez por linha física)
    # ========================================================
    #
    # [■] [ ] [ ] [■] [ ] [■]
    #  └─────────┘     └────┘
    #
    # O boundary é considerado o FINAL do chunk. Conectores só
    # são desenhados dentro da mesma linha física do grid.
    # ========================================================

    for row in range(n_rows_used):
        row_start = row * grid_cols
        row_end = min(row_start + grid_cols, n_draw)

        y_top = -row * row_height
        y_line = y_top + off_line
        y_line_top = y_top + off_line_top

        boundary_cols = [
            (i - row_start)
            for i in range(row_start, row_end)
            if inds[i] >= 0.5
        ]

        if not boundary_cols:
            # A linha inteira faz parte de um chunk que nem começa nem
            # termina aqui (veio de antes e continua depois) — desenha
            # uma linha reta atravessando toda a largura da linha.
            row_len = row_end - row_start
            x_start = 0.0 + connector_gap
            x_end = float(row_len - 1) - connector_gap
            if x_end > x_start:
                ax.plot([x_start, x_end], [y_line, y_line], color="blue", linewidth=1.5)
            continue

        # Seta de entrada: se o primeiro boundary da linha não está na
        # coluna 0, o chunk já vinha de antes (linha/wrap anterior) —
        # indica isso com uma seta entrando pela esquerda até o boundary.
        if boundary_cols[0] > 0:
            first_boundary = float(boundary_cols[0]) - connector_gap
            arrow_start = min(first_boundary - 1.0, -0.2)
            arrow_start = max(arrow_start, -0.8)

            if first_boundary > arrow_start:
                ax.annotate(
                    "",
                    xy=(first_boundary, y_line),
                    xytext=(arrow_start, y_line),
                    arrowprops=dict(arrowstyle="->", color="blue", linewidth=1.5),
                )

        for start, end in zip(boundary_cols[:-1], boundary_cols[1:]):
            x_start = float(start) + connector_gap
            x_end = float(end) - connector_gap

            # Se os dois boundaries são adjacentes (ou quase), não sobra
            # espaço pro "U" com gap — pula, evita linha invertida/cruzada.
            if x_end <= x_start:
                continue

            ax.plot([x_start, x_end], [y_line, y_line], color="blue", linewidth=1.5)
            ax.plot([x_start, x_start], [y_line, y_line_top], color="blue", linewidth=1.5)
            ax.plot([x_end, x_end], [y_line, y_line_top], color="blue", linewidth=1.5)

        # Seta depois do último boundary da linha, também com gap em
        # relação à caixa, até o fim do conteúdo daquela linha
        last_boundary = float(boundary_cols[-1]) + connector_gap
        row_content_end = float(row_end - row_start - 1)
        arrow_end = max(last_boundary + 1.0, row_content_end + 0.2)
        arrow_end = min(arrow_end, grid_cols - 0.2)

        if arrow_end > last_boundary:
            ax.annotate(
                "",
                xy=(arrow_end, y_line),
                xytext=(last_boundary, y_line),
                arrowprops=dict(arrowstyle="->", color="blue", linewidth=1.5),
            )

    # --------------------------------------------------------
    # Camada extra: boundaries do PRÓXIMO estágio, mapeados de volta
    # para estas colunas (chunks do stage seguinte agrupam vários
    # chunks deste stage). Usa o `ind` ORIGINAL (não o com índice 0
    # forçado) para reconstruir os chunk_spans reais — forçar o índice
    # 0 é só cosmético para a caixa verde, não deve afetar a contagem
    # de chunks nem o mapeamento entre estágios.
    # --------------------------------------------------------

    if child_ind is not None:
        raw_ind = data["ind"]  # sem o override de inds[0] = 1
        chunk_spans = compute_chunk_spans(raw_ind[:n])

        try:
            parent_ranges = map_boundaries_to_parent_columns(chunk_spans, child_ind)
        except ValueError as e:
            print(f"[draw_stage] Aviso ao mapear stage {stage} -> "
                  f"stage {child_stage_label}: {e}")
            parent_ranges = []

        if parent_ranges:
            draw_connector_layer(
                ax,
                parent_ranges,
                grid_cols=grid_cols,
                row_height=row_height,
                y_line_offset=off_line - 0.30,
                y_tick_offset=off_line_top - 0.30,
                color="red",
                connector_gap=connector_gap,
            )

    # --------------------------------------------------------
    # Rótulo "Stage N" (título da linha, não mais ao lado)
    # --------------------------------------------------------

    title = f"Stage {stage}"
    if truncated:
        title += f"  (mostrando {n_draw} de {n} — truncado em {MAX_POSITIONS})"
    if child_ind is not None:
        title += f"   [vermelho: chunks do Stage {child_stage_label}]"

    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=8)

    # --------------------------------------------------------
    # Configuração do eixo
    # --------------------------------------------------------

    extra_bottom = 0.30 if child_ind is not None else 0.0
    ax.set_xlim(-1, grid_cols + 1)
    ax.set_ylim(-(n_rows_used - 1) * row_height - 0.55 - extra_bottom, 1.15)
    ax.axis("off")


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
    Cria:

        text/output/<model>/frase_<phrase>.gif

    a partir de:

        text/output/<model>/<phrase>/step_N.txt
    """

    # --------------------------------------------------------
    # Diretório de entrada
    # --------------------------------------------------------

    if output_dir is None:
        output_dir = (
            Path("text")
            / "output"
            / model
            / str(phrase)
        )

    else:
        output_dir = Path(output_dir)

    # --------------------------------------------------------
    # Arquivos
    # --------------------------------------------------------

    step_files = find_step_files(output_dir)

    if not step_files:
        raise RuntimeError(
            f"Nenhum arquivo step_N.txt encontrado em:\n"
            f"{output_dir}"
        )

    print(f"Encontrados {len(step_files)} steps")

    # --------------------------------------------------------
    # Carrega todos os frames
    # --------------------------------------------------------

    frames = []

    for step, path in step_files:

        stages = read_step_file(path)

        if not stages:
            print(
                f"Aviso: nenhum Stage encontrado em {path}"
            )
            continue

        frames.append(
            {
                "step": step,
                "stages": stages,
            }
        )

    if not frames:
        raise RuntimeError(
            "Nenhum frame válido foi encontrado."
        )

    # --------------------------------------------------------
    # Número de stages
    # --------------------------------------------------------

    num_stages = max(
        len(frame["stages"])
        for frame in frames
    )

    # --------------------------------------------------------
    # Linhas de grid necessárias por stage (baseado no maior
    # comprimento visto em qualquer frame daquele stage — o
    # texto é o mesmo em todos os passos, então isso é estável)
    # --------------------------------------------------------

    stage_rows = {}
    any_truncated = False

    for frame in frames:
        for stage_idx, stage_data in frame["stages"].items():
            n = len(stage_data["char"])
            if n > MAX_POSITIONS:
                any_truncated = True
            r = rows_needed(n)
            stage_rows[stage_idx] = max(stage_rows.get(stage_idx, 1), r)

    if any_truncated:
        print(
            f"Aviso: algum stage excede {MAX_POSITIONS} posições "
            f"(grid {GRID_ROWS}x{GRID_COLS}) — será truncado nos frames."
        )

    max_rows_used = max(stage_rows.values()) if stage_rows else 1

    # --------------------------------------------------------
    # Tamanho da figura
    #
    # Largura fixa (grid_cols é sempre 40, não depende mais do
    # texto). Altura por stage é proporcional às linhas de grid
    # que ele de fato usa; todos os painéis usam a mesma escala
    # vertical (max_rows_used) para simplicidade e para o layout
    # não mudar de proporção entre frames.
    # --------------------------------------------------------

    fig_width = 16.0

    row_height_in = 0.42  # polegadas por linha física do grid
    stage_header_in = 0.55  # espaço para o título "Stage N"

    fig_height = 1.0 + num_stages * (
        stage_header_in + max_rows_used * row_height_in
    )

    fig, axes = plt.subplots(
        num_stages,
        1,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    axes = axes.flatten()

    # ========================================================
    # UPDATE
    # ========================================================

    def update(frame_index: int) -> Iterable[Artist]:

        frame = frames[frame_index]

        # ----------------------------------------------------
        # Limpa os stages anteriores
        # ----------------------------------------------------

        for ax in axes:
            ax.clear()

        # ----------------------------------------------------
        # Desenha cada Stage
        # ----------------------------------------------------

        for stage_idx in range(num_stages):

            ax = axes[stage_idx]

            if stage_idx not in frame["stages"]:
                ax.axis("off")
                continue

            next_stage_idx = stage_idx + 1
            child_ind = None
            if next_stage_idx in frame["stages"]:
                child_ind = frame["stages"][next_stage_idx]["ind"]

            draw_stage(
                ax=ax,
                stage=stage_idx,
                data=frame["stages"][stage_idx],
                grid_cols=GRID_COLS,
                grid_rows=GRID_ROWS,
                child_ind=child_ind,
                child_stage_label=next_stage_idx,
            )

        # ----------------------------------------------------
        # Step atual
        # ----------------------------------------------------

        fig.suptitle(
            f"{model} — frase {phrase} — Step {frame['step']}",
            fontsize=14,
            fontweight="bold",
        )

        fig.subplots_adjust(
            left=0.03,
            right=0.98,
            top=0.90,
            bottom=0.03,
            hspace=0.55,
        )

        # ----------------------------------------------------
        # Necessário para satisfazer a assinatura esperada
        # pelo Matplotlib/Pylance.
        # ----------------------------------------------------

        return tuple(
            artist
            for ax in axes
            for artist in ax.get_children()
        )

    # ========================================================
    # ANIMATION
    # ========================================================

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        repeat=True,
        blit=False,
    )

    # --------------------------------------------------------
    # Arquivo de saída
    # --------------------------------------------------------

    if output_file is None:

        output_file = (
            Path("text")
            / "output"
            / model
            / f"frase_{phrase}.gif"
        )

    else:
        output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Salvando GIF em:")
    print(output_file)

    # --------------------------------------------------------
    # Salva
    # --------------------------------------------------------

    animation.save(
        output_file,
        writer=PillowWriter(fps=fps),
    )

    plt.close(fig)

    print("GIF criado com sucesso.")


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Gera GIF da evolução dos boundaries do H-Net."
    )

    parser.add_argument(
        "model",
        help="Nome do modelo",
    )

    parser.add_argument(
        "phrase",
        type=int,
        choices=range(1, 5),
        help="Número da frase (1-4)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames por segundo",
    )

    parser.add_argument(
        "--input-dir",
        default=None,
        help="Diretório contendo os step_N.txt",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Arquivo GIF de saída",
    )

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