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

Entradas aceitas (podem ser misturadas na mesma chamada):
  * um JSON com um objeto {meta, text, results};
  * um JSON com uma LISTA de objetos (saída do --n-samples);
  * um .jsonl (um objeto por linha);
  * vários arquivos de qualquer um dos tipos acima.
Todas as amostras são agregadas. Se um tokenizador falhou em algumas amostras,
essas amostras só são ignoradas para ele (e para os pares em que ele entra).

Uso:
    python compare_tokenizers_stats.py dados.json
    python compare_tokenizers_stats.py amostras.json -o saida.png
    python compare_tokenizers_stats.py a.json b.json dados.jsonl --exclude bytes
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
        samples.extend(o for o in objs if isinstance(o, dict) and "text" in o and "results" in o)
    if not samples:
        raise SystemExit("nenhuma amostra {text, results} encontrada nos arquivos informados")
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
        }
    return out


# --------------------------------------------------------------------------
# Estatísticas agregadas
# --------------------------------------------------------------------------
def aggregate(samples, exclude):
    analyzed = [analyze_sample(s, set(exclude)) for s in samples]

    # união ordenada dos tokenizadores (um pode ter falhado em alguma amostra)
    names = []
    for a in analyzed:
        for nm in a["tok"]:
            if nm not in names:
                names.append(nm)
    if not names:
        raise SystemExit("nenhum tokenizador nas amostras (--exclude removeu todos?)")
    k = len(names)
    idx = {nm: i for i, nm in enumerate(names)}

    inter = np.zeros((k, k)); union = np.zeros((k, k))
    s_inter = np.zeros((k, k)); s_union = np.zeros((k, k))
    both = np.zeros((k, k)); a_only = np.zeros((k, k))
    b_only = np.zeros((k, k)); neither = np.zeros((k, k))
    tokens = np.zeros(k); skipped = np.zeros(k)
    words = np.zeros(k); chars = np.zeros(k); n_samp = np.zeros(k, dtype=int)

    for a in analyzed:
        cand = a["intra"]
        for nm, t in a["tok"].items():
            i = idx[nm]
            tokens[i] += t["n_tokens"]
            skipped[i] += t["skipped"]
            words[i] += a["words"]
            chars[i] += a["chars"]
            n_samp[i] += 1
        for x, y in combinations([nm for nm in names if nm in a["tok"]], 2):
            i, j = idx[x], idx[y]
            A, B = a["tok"][x]["intra_cuts"], a["tok"][y]["intra_cuts"]
            SA, SB = a["tok"][x]["spans"], a["tok"][y]["spans"]
            nb = len(A & B); na = len(A - B); nbo = len(B - A)
            nn = len(cand) - nb - na - nbo
            for m, v in ((inter, nb), (union, len(A | B)),
                         (s_inter, len(SA & SB)), (s_union, len(SA | SB)),
                         (both, nb), (neither, nn)):
                m[i, j] += v; m[j, i] += v
            a_only[i, j] += na; a_only[j, i] += nbo   # só em A (linha) / só em B (coluna)
            b_only[i, j] += nbo; b_only[j, i] += na

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
        "tokens_per_word": safe_div(tokens, words, np.nan),
        "chars_per_token": safe_div(chars, tokens, np.nan),
        "skipped": skipped, "n_samples": len(analyzed), "n_samples_tok": n_samp,
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
    print(f"{'tokenizador':<20}{'amostras':>10}{'tokens/palavra':>16}"
          f"{'chars/token':>14}{'peças ignoradas':>18}")
    for i, n in enumerate(names):
        print(f"{n:<20}{st['n_samples_tok'][i]:>10}{st['tokens_per_word'][i]:>16.2f}"
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