#!/usr/bin/env python3
"""Compara estatisticamente o quanto os tokenizadores concordam nos cortes.

Cada tokenizador é reduzido ao conjunto de posições de corte no texto (sem
espaços). A partir daí são calculadas, para cada par de tokenizadores:

  * Jaccard dos cortes *dentro de palavras* (cortes entre palavras são
    triviais: todo mundo concorda, então inflariam a similaridade);
  * Kappa de Cohen sobre as posições candidatas dentro de palavras
    (corrige a concordância esperada ao acaso);
  * Jaccard de spans: fração de tokens idênticos (mesmo início e fim).

Também mostra fertilidade (tokens por palavra) e caracteres por token.

Uso:
    python compare_tokenizers.py dados.json
    python compare_tokenizers.py amostra1.json amostra2.json ... -o saida.png
    python compare_tokenizers.py dados.jsonl --exclude bytes

Com vários JSONs (ou um .jsonl) os contadores são agregados, o que dá
estatísticas bem mais estáveis do que uma única frase.
"""
import argparse
import json
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------
# Leitura
# --------------------------------------------------------------------------
def load_samples(paths):
    samples = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        try:
            obj = json.loads(raw)
            objs = obj if isinstance(obj, list) else [obj]
        except json.JSONDecodeError:  # tenta JSONL
            objs = [json.loads(line) for line in raw.splitlines() if line.strip()]
        samples.extend(objs)
    return samples


# --------------------------------------------------------------------------
# Alinhamento das peças ao texto original
# --------------------------------------------------------------------------
def align_starts(text, pieces):
    """Devolve (offsets de início de cada peça alinhada, nº de peças ignoradas).

    Remove marcadores de espaço (▁, espaço inicial) e de continuação (##) e
    procura a peça no texto a partir da posição atual, pulando espaços.
    Peças que não casam (ex.: bytes parciais de UTF-8 decodificados
    isoladamente) são ignoradas sem desalinhar as seguintes.
    """
    starts, skipped, pos = [], 0, 0
    for p in pieces:
        q = p.replace("▁", " ").strip()
        if q.startswith("##"):
            q = q[2:]
        if not q:
            continue
        j = pos
        while j < len(text) and text[j].isspace():
            j += 1
        if text.startswith(q, j):
            starts.append(j)
            pos = j + len(q)
        else:
            skipped += 1
    return starts, skipped


def analyze_sample(sample, names_filter):
    """Converte um sample em conjuntos de cortes por tokenizador."""
    text = sample["text"]
    ns = [i for i, c in enumerate(text) if not c.isspace()]  # índices não-espaço
    comp = {orig: k for k, orig in enumerate(ns)}             # orig -> compacto
    n = len(ns)

    cand_intra = {k for k in range(1, n) if ns[k] == ns[k - 1] + 1}
    word_cuts = set(range(1, n)) - cand_intra

    out = {"n": n, "intra": cand_intra, "words": len(text.split()),
           "chars": len(text), "tok": {}}
    for r in sample["results"]:
        name = r["tokenizer"]
        if name in names_filter:
            continue
        starts, skipped = align_starts(text, r["pieces"])
        cuts = {comp[s] for s in starts if comp[s] > 0}
        full = sorted(cuts | word_cuts)
        edges = [0] + full + [n]
        spans = set(zip(edges[:-1], edges[1:]))
        out["tok"][name] = {
            "intra_cuts": cuts & cand_intra,
            "spans": spans,
            "n_tokens": r["n_tokens"],
            "skipped": skipped,
            "aligned_chars": sum(len(p.replace('▁', ' ').strip().lstrip('#')) for p in r["pieces"]),
        }
    return out


# --------------------------------------------------------------------------
# Estatísticas agregadas
# --------------------------------------------------------------------------
def aggregate(samples, exclude):
    analyzed = [analyze_sample(s, set(exclude)) for s in samples]
    names = list(analyzed[0]["tok"].keys())
    k = len(names)
    idx = {nm: i for i, nm in enumerate(names)}

    inter = np.zeros((k, k)); union = np.zeros((k, k))
    s_inter = np.zeros((k, k)); s_union = np.zeros((k, k))
    both = np.zeros((k, k)); a_only = np.zeros((k, k))
    b_only = np.zeros((k, k)); neither = np.zeros((k, k))
    tokens = np.zeros(k); skipped = np.zeros(k)
    total_words = 0; total_chars = 0

    for a in analyzed:
        total_words += a["words"]; total_chars += a["chars"]
        cand = a["intra"]
        for nm in names:
            tokens[idx[nm]] += a["tok"][nm]["n_tokens"]
            skipped[idx[nm]] += a["tok"][nm]["skipped"]
        for x, y in combinations(names, 2):
            i, j = idx[x], idx[y]
            A, B = a["tok"][x]["intra_cuts"], a["tok"][y]["intra_cuts"]
            SA, SB = a["tok"][x]["spans"], a["tok"][y]["spans"]
            nb = len(A & B); na = len(A - B); nbo = len(B - A)
            nn = len(cand) - nb - na - nbo
            for m, v in ((inter, nb), (union, len(A | B)),
                         (s_inter, len(SA & SB)), (s_union, len(SA | SB)),
                         (both, nb), (a_only, na), (b_only, nbo), (neither, nn)):
                m[i, j] += v; m[j, i] += v if m is not a_only and m is not b_only else 0
            a_only[j, i] += nbo; b_only[j, i] += na  # simetria correta

    def safe_div(num, den, default=1.0):
        return np.where(den > 0, num / np.where(den > 0, den, 1), default)

    jac = safe_div(inter, union)
    span_jac = safe_div(s_inter, s_union)

    N = both + a_only + b_only + neither
    po = safe_div(both + neither, N)
    pa = safe_div(both + a_only, N, 0.0)
    pb = safe_div(both + b_only, N, 0.0)
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = np.where(pe < 1, (po - pe) / np.where(pe < 1, 1 - pe, 1), np.nan)

    for m in (jac, span_jac, kappa):
        np.fill_diagonal(m, 1.0)

    return {
        "names": names, "jaccard": jac, "span": span_jac, "kappa": kappa,
        "tokens_per_word": tokens / total_words,
        "chars_per_token": total_chars / tokens,
        "skipped": skipped, "n_samples": len(analyzed),
    }


# --------------------------------------------------------------------------
# Saída
# --------------------------------------------------------------------------
def print_matrix(title, names, m):
    print(f"\n{title}")
    w = max(len(n) for n in names) + 2
    print(" " * w + "".join(f"{n:>{w}}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:<{w}}" + "".join(f"{m[i, j]:>{w}.3f}" for j in range(len(names))))


def heatmap(ax, m, names, title):
    im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(len(names)):
        for j in range(len(names)):
            v = m[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v < 0.6 else "black")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_paths", nargs="+")
    ap.add_argument("-o", "--output", help="salva a figura (png/pdf/svg)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="tokenizadores a ignorar (ex.: bytes)")
    args = ap.parse_args()

    samples = load_samples(args.json_paths)
    st = aggregate(samples, args.exclude)
    names = st["names"]

    print(f"{st['n_samples']} amostra(s), tokenizadores: {', '.join(names)}")
    print("\nResumo por tokenizador")
    print(f"{'tokenizador':<15}{'tokens/palavra':>16}{'chars/token':>14}{'peças ignoradas':>18}")
    for i, n in enumerate(names):
        print(f"{n:<15}{st['tokens_per_word'][i]:>16.2f}"
              f"{st['chars_per_token'][i]:>14.2f}{int(st['skipped'][i]):>18}")

    print_matrix("Jaccard dos cortes dentro de palavras", names, st["jaccard"])
    print_matrix("Kappa de Cohen (cortes dentro de palavras)", names, st["kappa"])
    print_matrix("Jaccard de spans (tokens idênticos)", names, st["span"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    heatmap(axes[0, 0], st["jaccard"], names, "Jaccard — cortes dentro de palavras")
    im = heatmap(axes[0, 1], st["kappa"], names, "Kappa de Cohen — cortes dentro de palavras")
    heatmap(axes[1, 0], st["span"], names, "Jaccard — tokens idênticos (span)")
    fig.colorbar(im, ax=axes[:2, :].ravel().tolist(), shrink=0.6, label="similaridade")

    ax = axes[1, 1]
    x = np.arange(len(names))
    ax.bar(x, st["tokens_per_word"], color="#4c8bf5")
    ax.set_xticks(x, names, rotation=45, ha="right")
    ax.set_ylabel("tokens por palavra (fertilidade)")
    ax.set_title("Fertilidade", fontsize=11, fontweight="bold")
    for xi, v in zip(x, st["tokens_per_word"]):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Similaridade entre tokenizações ({st['n_samples']} amostra(s))",
                 fontsize=13)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"\nFigura salva em {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
