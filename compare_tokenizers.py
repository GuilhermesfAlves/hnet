"""
compare_tokenizers.py — mostra como diferentes tokenizadores segmentam o MESMO texto,
sorteado de um split de um dataset do HF Hub (mesmos parâmetros de dataset do eval_pt.py).

Tokenizadores (cada um é opcional; se a dependência faltar, é ignorado com aviso):
  bpe        BPE byte-level (GPT-2)             -> transformers  (--bpe-name)
  tiktoken   BPE byte-level (cl100k_base)       -> tiktoken      (--tiktoken-name)
  wordpiece  WordPiece (BERT multilíngue)       -> transformers  (--wordpiece-name)
  sentencepiece  Unigram/SentencePiece (XLM-R)  -> transformers  (--sp-name)
  spacy      tokenização por regras (palavras)  -> spacy         (--spacy-model / --spacy-lang)
  bytes      bytes UTF-8 (a "tokenização" do H-Net)

Instalação:
  pip install datasets transformers tiktoken spacy sentencepiece protobuf python-dotenv
  python -m spacy download pt_core_news_sm      # opcional (sem ele usa spacy.blank)

Exemplos:
  python compare_tokenizers.py --dataset-name uonlp/CulturaX --config-name pt --split train
  python compare_tokenizers.py --dataset-name org/ds --split validation --text-column content \
      --seed 7 --max-chars 400 --tokenizers bpe,wordpiece,spacy,bytes
  python compare_tokenizers.py ... --json-out amostra.json     # salva texto + tokens
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from itertools import islice

from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

SEP = "│"


# --------------------------------------------------------------------------- #
# Amostra aleatória do dataset
# --------------------------------------------------------------------------- #
def sample_text(args) -> tuple[str, dict]:
    from datasets import load_dataset

    ds = load_dataset(
        args.dataset_name,
        args.config_name,
        split=args.split,
        streaming=args.streaming,
        token=HF_TOKEN,
        trust_remote_code=args.trust_remote_code,
    )
    meta = {
        "dataset": args.dataset_name, 
        "config": args.config_name,
        "split": args.split, 
        "seed": args.seed
    }

    if args.streaming:
        # embaralha por buffer (determinístico com a seed) e pega o 1º texto longo o bastante
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
        for i, ex in enumerate(islice(ds, args.max_scan)):
            txt = ex.get(args.text_column)
            if isinstance(txt, str) and len(txt.strip()) >= args.min_chars:
                meta["stream_pos"] = i
                return txt, meta
        raise RuntimeError(
            f"nenhum texto com >= {args.min_chars} chars nos primeiros {args.max_scan} exemplos")

    rng = random.Random(args.seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    for i in idxs[: args.max_scan]:
        txt = ds[i][args.text_column]
        if isinstance(txt, str) and len(txt.strip()) >= args.min_chars:
            meta["index"] = i
            return txt, meta
    raise RuntimeError(f"nenhum texto com >= {args.min_chars} chars encontrado")


def crop(text: str, max_chars: int, rng: random.Random) -> str:
    """Recorta um trecho aleatório (começando em fronteira de palavra) de até max_chars."""
    text = " ".join(text.split())          # normaliza quebras de linha p/ exibição
    if len(text) <= max_chars:
        return text
    start = rng.randint(0, len(text) - max_chars)
    while start > 0 and text[start - 1] != " ":
        start -= 1
    return text[start:start + max_chars].strip()


# --------------------------------------------------------------------------- #
# Tokenizadores — cada um devolve lista de strings (pedaços do texto) + ids opcionais
# --------------------------------------------------------------------------- #
def _hf_offsets(name: str, text: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, token=HF_TOKEN, use_fast=True)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    raw = tok.convert_ids_to_tokens(ids)
    # pedaços do texto original (via offsets) — legíveis, sem Ġ / ## / ▁
    pieces = [text[a:b] for a, b in enc["offset_mapping"]]
    return pieces, raw, ids, tok.vocab_size


def tok_bpe(text, args):
    return _hf_offsets(args.bpe_name, text)


def tok_wordpiece(text, args):
    return _hf_offsets(args.wordpiece_name, text)


def tok_sentencepiece(text, args):
    return _hf_offsets(args.sp_name, text)


def tok_tiktoken(text, args):
    import tiktoken

    enc = tiktoken.get_encoding(args.tiktoken_name)
    ids = enc.encode(text)
    bs = enc.decode_tokens_bytes(ids)
    pieces = [b.decode("utf-8", errors="replace") for b in bs]   # byte parcial vira �
    raw = [repr(b)[2:-1] for b in bs]
    return pieces, raw, ids, enc.n_vocab


def tok_spacy(text, args):
    import spacy

    try:
        nlp = spacy.load(args.spacy_model, disable=["parser", "ner", "lemmatizer"])
    except OSError:
        print(f"  (modelo spaCy '{args.spacy_model}' ausente -> spacy.blank('{args.spacy_lang}'))",
              file=sys.stderr)
        nlp = spacy.blank(args.spacy_lang)
    doc = nlp.make_doc(text)                 # só tokeniza (rápido)
    pieces = [t.text for t in doc if not t.is_space]
    return pieces, pieces, None, None


def tok_bytes(text, args):
    bs = text.encode("utf-8")
    # agrupa os bytes de cada caractere, para exibição legível (1 byte = 1 token no H-Net)
    pieces, raw = [], []
    for ch in text:
        b = ch.encode("utf-8")
        pieces.append(ch)
        raw.append(" ".join(f"{x:02x}" for x in b))
    return pieces, raw, list(bs), 256


TOKENIZERS = {
    "bpe": ("BPE byte-level (GPT-2)", tok_bpe),
    "tiktoken": ("BPE byte-level (tiktoken)", tok_tiktoken),
    "wordpiece": ("WordPiece (BERT)", tok_wordpiece),
    "sentencepiece": ("SentencePiece Unigram (XLM-R)", tok_sentencepiece),
    "spacy": ("spaCy (regras)", tok_spacy),
    "bytes": ("Bytes UTF-8 (H-Net)", tok_bytes),
}


# --------------------------------------------------------------------------- #
# Exibição
# --------------------------------------------------------------------------- #
def visible(p: str) -> str:
    return p.replace("\n", "\\n").replace("\t", "\\t").replace(" ", "·") or "∅"


def show(key, label, text, result, args) -> dict:
    pieces, raw, ids, vocab = result
    n = len(pieces)
    n_bytes = len(text.encode("utf-8"))
    print("\n" + "=" * 100)
    print(f"{label}  [{key}]" + (f"  | vocab={vocab:,}" if vocab else ""))
    print(f"tokens: {n} | chars/token: {len(text) / max(n, 1):.2f} | "
          f"bytes/token: {n_bytes / max(n, 1):.2f}")
    print("-" * 100)
    if key == "bytes":
        shown = pieces[: args.max_show]
        print(SEP.join(visible(p) for p in shown) + (f"{SEP}…" if n > len(shown) else ""))
    else:
        print(SEP.join(visible(p) for p in pieces[: args.max_show])
              + (f"{SEP}…" if n > args.max_show else ""))
    if args.show_ids and ids is not None:
        print("ids:", ids[: args.max_show], "…" if n > args.max_show else "")
    if args.show_raw and key not in ("spacy",):
        print("raw:", raw[: args.max_show], "…" if n > args.max_show else "")
    return {"tokenizer": key, "n_tokens": n, "chars_per_token": len(text) / max(n, 1),
            "bytes_per_token": n_bytes / max(n, 1), "pieces": pieces, "raw": raw, "ids": ids}


def parse_args():
    p = argparse.ArgumentParser(description="Compara tokenizadores no mesmo texto sorteado.")
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--config-name", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--text-column", default="text")
    p.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trust-remote-code", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-buffer", type=int, default=1000)
    p.add_argument("--max-scan", type=int, default=2000,
                   help="máximo de exemplos examinados até achar um texto válido")
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=300,
                   help="tamanho do trecho tokenizado (recorte aleatório do texto sorteado)")

    p.add_argument("--tokenizers", default="bpe,tiktoken,wordpiece,sentencepiece,spacy,bytes")
    p.add_argument("--bpe-name", default="gpt2")
    p.add_argument("--tiktoken-name", default="cl100k_base")
    p.add_argument("--wordpiece-name", default="bert-base-multilingual-cased")
    p.add_argument("--sp-name", default="xlm-roberta-base")
    p.add_argument("--spacy-model", default="pt_core_news_sm")
    p.add_argument("--spacy-lang", default="pt", help="idioma do spacy.blank (fallback)")

    p.add_argument("--max-show", type=int, default=120, help="máx. de tokens exibidos")
    p.add_argument("--show-ids", action="store_true", default=False)
    p.add_argument("--show-raw", action="store_true", default=False,
                   help="mostra os tokens crus (Ġ, ##, ▁, bytes)")
    p.add_argument("--json-out", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    keys = [k.strip() for k in args.tokenizers.split(",") if k.strip()]
    bad = [k for k in keys if k not in TOKENIZERS]
    if bad:
        print(f"tokenizadores desconhecidos: {bad}; válidos: {list(TOKENIZERS)}")
        return 2

    full, meta = sample_text(args)
    text = crop(full, args.max_chars, random.Random(args.seed))
    meta.update(n_chars=len(text), n_bytes=len(text.encode("utf-8")))

    print(f"Dataset: {args.dataset_name}"
          f"{'/' + args.config_name if args.config_name else ''} [{args.split}] | seed={args.seed}")
    print(f"Texto ({len(text)} chars, {meta['n_bytes']} bytes):\n")
    print(text)
    print("\n(legenda: '·' = espaço, '│' = fronteira entre tokens, '∅' = token vazio)")

    results = []
    for k in keys:
        label, fn = TOKENIZERS[k]
        try:
            results.append(show(k, label, text, fn(text, args), args))
        except ImportError as e:
            print(f"\n[{k}] ignorado — dependência ausente: {e}")
        except Exception as e:  # download/modelo indisponível etc.
            print(f"\n[{k}] erro: {type(e).__name__}: {e}")

    if results:
        print("\n" + "=" * 100)
        print(f"{'tokenizador':<16}{'tokens':>8}{'chars/tok':>12}{'bytes/tok':>12}")
        for r in sorted(results, key=lambda r: r["n_tokens"]):
            print(f"{r['tokenizer']:<16}{r['n_tokens']:>8}"
                  f"{r['chars_per_token']:>12.2f}{r['bytes_per_token']:>12.2f}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "text": text, "results": results}, f,
                      ensure_ascii=False, indent=2)
        print(f"\nSalvo em {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
