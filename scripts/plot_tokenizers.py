#!/usr/bin/env python3
"""Plota como cada tokenizador separou a frase de um JSON de comparação.

Uso:
    python plot_tokenizers.py dados.json                # abre a janela
    python plot_tokenizers.py dados.json -o saida.png   # salva em arquivo
"""
import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

LINE_WIDTH = 110   # largura máxima de cada linha (em "caracteres")
PAD = 0.8          # espaço extra por caixa
PALETTE = ["#cfe8ff", "#ffe3b8", "#d6f5d6", "#f7d1e8", "#e6dcff", "#fff4b3"]


def visible(piece: str) -> str:
    """Deixa espaços visíveis dentro das caixas."""
    return piece.replace(" ", "·").replace("\n", "↵")


def wrap(pieces):
    """Quebra a lista de peças em linhas de até LINE_WIDTH."""
    lines, cur, w = [], [], 0.0
    for p in pieces:
        bw = max(len(p), 1) + PAD
        if cur and w + bw > LINE_WIDTH:
            lines.append(cur)
            cur, w = [], 0.0
        cur.append((p, bw))
        w += bw
    if cur:
        lines.append(cur)
    return lines


def draw(ax, pieces):
    lines = wrap(pieces)
    ax.set_xlim(0, LINE_WIDTH)
    ax.set_ylim(len(lines), 0)
    ax.axis("off")
    idx = 0
    for row, line in enumerate(lines):
        x = 0.0
        for piece, bw in line:
            ax.add_patch(
                Rectangle((x, row + 0.1), bw, 0.8,
                          facecolor=PALETTE[idx % len(PALETTE)],
                          edgecolor="#555", linewidth=0.6)
            )
            ax.text(x + bw / 2, row + 0.5, visible(piece),
                    ha="center", va="center", fontsize=8, family="monospace")
            x += bw
            idx += 1
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("-o", "--output", help="salva a figura (png/pdf/svg)")
    args = ap.parse_args()

    with open(args.json_path, encoding="utf-8") as f:
        data = json.load(f)

    text = data["text"]
    results = data["results"]

    # pré-calcula as linhas para dimensionar a altura de cada subplot
    n_lines = [len(wrap(r["pieces"])) for r in results]
    fig = plt.figure(figsize=(20, 1.0 + sum(0.55 * n + 0.6 for n in n_lines)))
    gs = GridSpec(len(results), 1, height_ratios=n_lines, hspace=0.6, figure=fig)

    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[i])
        draw(ax, r["pieces"])
        ax.set_title(
            f'{r["tokenizer"]}  —  {r["n_tokens"]} tokens  |  '
            f'{r["chars_per_token"]:.2f} chars/token  |  '
            f'{r["bytes_per_token"]:.2f} bytes/token',
            loc="left", fontsize=11, fontweight="bold",
        )

    preview = text if len(text) <= 160 else text[:157] + "..."
    fig.suptitle(f"Tokenização da frase:\n{preview}", fontsize=12, y=0.995)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Figura salva em {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
