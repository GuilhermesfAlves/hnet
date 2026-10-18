"""
compare_tokenizers.py — mostra como diferentes tokenizadores segmentam o MESMO texto,
sorteado de um split de um dataset do HF Hub (mesmos parâmetros de dataset do eval_pt.py).

Tokenizadores (cada um é opcional; se a dependência faltar, é ignorado com aviso):
  bpe        BPE byte-level (GPT-2)             -> transformers  (--bpe-name)
  tiktoken   BPE byte-level (cl100k_base)       -> tiktoken      (--tiktoken-name)
  wordpiece  WordPiece (BERT multilíngue)       -> transformers  (--wordpiece-name)
  sentencepiece  Unigram/SentencePiece (XLM-R)  -> transformers  (--sp-name)
  spacy      tokenização por regras (palavras)  -> spacy         (--spacy-model / --spacy-lang)
  bytes      bytes UTF-8
  hnet_1stage_L       H-Net 1 estágio (L)       -> checkpoint    (--hnet-1stage-L-config/-ckpt)
  hnet_2stage_L       H-Net 2 estágios (L); chunks do ÚLTIMO estágio (mais grosso)
  hnet_2stage_L_s0    H-Net 2 estágios (L); chunks do 1º estágio (mais fino)
  hnet_1stage_Llama   H-Net 1 estágio (Llama)   -> checkpoint    (--hnet-1stage-Llama-config/-ckpt)

  Nos H-Nets, cada "token" é um chunk: o trecho entre duas posições onde
  output.bpred_output[s].boundary_mask é True. As chaves hnet_* não estão na lista
  padrão de --tokenizers; é preciso pedi-las explicitamente (e passar config + checkpoint).

Instalação:
  pip install datasets transformers tiktoken spacy sentencepiece protobuf python-dotenv
  python -m spacy download pt_core_news_sm      # opcional (sem ele usa spacy.blank)

Exemplos:
  python compare_tokenizers.py --dataset-name uonlp/CulturaX --config-name pt --split train
  python compare_tokenizers.py --dataset-name org/ds --split validation --text-column content \
      --seed 7 --max-chars 400 --tokenizers bpe,wordpiece,spacy,bytes
  python compare_tokenizers.py ... --json-out amostra.json     # salva texto + tokens
  python compare_tokenizers.py --dataset-name uonlp/CulturaX --config-name pt \
      --tokenizers bpe,hnet_1stage_L,hnet_2stage_L,hnet_2stage_L_s0,hnet_1stage_Llama \
      --hnet-1stage-L-config configs/hnet_1stage_L.json --hnet-1stage-L-ckpt ckpts/1stage_L.pt \
      --hnet-2stage-L-config configs/hnet_2stage_L.json --hnet-2stage-L-ckpt ckpts/2stage_L.pt \
      --hnet-1stage-Llama-config configs/hnet_1stage_Llama.json \
      --hnet-1stage-Llama-ckpt ckpts/1stage_Llama.pt --hnet-repo /caminho/do/repo
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
    # agrupa os bytes de cada caractere, para exibição legível
    pieces, raw = [], []
    for ch in text:
        b = ch.encode("utf-8")
        pieces.append(ch)
        raw.append(" ".join(f"{x:02x}" for x in b))
    return pieces, raw, list(bs), 256


# --------------------------------------------------------------------------- #
# H-Net (dynamic chunking): chunks = posições com boundary_mask == True
# --------------------------------------------------------------------------- #
def build_model(model_config_path: str, device: str, dtype):
    # >>> AJUSTE ESTES IMPORTS ao layout do seu repositório (ou use --hnet-repo) <<<
    from hnet.models.config_hnet import (
        AttnConfig, SSMConfig, HNetConfig, ExternalBackboneConfig,
    )
    from hnet.models.mixer_seq import HNetForCausalLM

    with open(model_config_path) as f:
        config = json.load(f)
    attn_cfg = AttnConfig(**config.pop("attn_cfg"))
    ssm_cfg  = SSMConfig(**config.pop("ssm_cfg"))
    external_backbones_raw = config.pop("external_backbones", {})
    external_backbones = {
        name: ExternalBackboneConfig(**cfg)
        for name, cfg in external_backbones_raw.items()
    }
    hnet_cfg = HNetConfig(**config, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg, external_backbones=external_backbones)
    model    = HNetForCausalLM(hnet_cfg, device=device, dtype=dtype)
    model.init_weights()
    return model, hnet_cfg


# chave -> rótulo. Os argumentos de cada um são --hnet-<sufixo>-config / --hnet-<sufixo>-ckpt
HNET_MODELS = {
    "hnet_1stage_L": "H-Net 1-stage (L)",
    "hnet_2stage_L": "H-Net 2-stage (L)",
    "hnet_1stage_Llama": "H-Net 1-stage (Llama)",
}

_STARTS_CACHE: dict = {}   # (chave, texto) -> posições de início de chunk por estágio


def _hnet_load(key: str, args):
    """Constrói o modelo com build_model e carrega o checkpoint de args.<key>_ckpt."""
    cfg_path = getattr(args, f"{key}_config")
    ckpt_path = getattr(args, f"{key}_ckpt")
    if not cfg_path or not ckpt_path:
        flag = key.replace("_", "-")
        raise ValueError(f"passe --{flag}-config e --{flag}-ckpt para usar '{key}'")

    import torch

    if args.hnet_repo and args.hnet_repo not in sys.path:
        sys.path.insert(0, args.hnet_repo)

    device = args.hnet_device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.hnet_dtype)
    model, _ = build_model(cfg_path, device, dtype)

    # weights_only=False: checkpoints de treino costumam ter objetos além de tensores
    # (carregue apenas checkpoints em que você confia).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    sd = {}
    for k, v in ckpt.items():
        for prefix in ("module.", "_orig_mod."):
            while k.startswith(prefix):
                k = k[len(prefix):]
        sd[k] = v
    res = model.load_state_dict(sd, strict=False)
    if res.missing_keys or res.unexpected_keys:
        print(f"  [{key}] aviso ao carregar checkpoint: "
              f"{len(res.missing_keys)} chaves faltando {res.missing_keys[:5]}, "
              f"{len(res.unexpected_keys)} inesperadas {res.unexpected_keys[:5]}",
              file=sys.stderr)
    model.eval()
    return model, device


def compose_stage_starts(masks, seq_len: int, offset: int = 0):
    """Converte as boundary_mask de cada estágio em posições de byte (início de chunk).

    masks[s] é uma lista de bool com um item por posição da sequência do estágio s.
    O estágio s+1 opera só sobre as posições escolhidas no estágio s, então as máscaras
    são compostas para voltar às coordenadas da sequência de entrada. `offset` é o nº
    de tokens extras no início (ex.: 1 se há BOS), descontado do resultado.
    Sempre inclui a posição 0 (início do texto).
    """
    pos = list(range(seq_len))
    out = []
    for m in masks:
        n = min(len(m), len(pos))
        sel = [p for p, flag in zip(pos[:n], m[:n]) if flag]
        out.append(sorted({p - offset for p in sel if p - offset >= 0} | {0}))
        pos = sel
    return out


def hnet_stage_starts(key: str, text: str, args):
    """Roda o H-Net sobre os bytes do texto e devolve, por estágio, os bytes onde começa um chunk."""
    cache_key = (key, text)
    if cache_key in _STARTS_CACHE:
        return _STARTS_CACHE[cache_key]

    import torch

    model, device = _hnet_load(key, args)
    ids = list(text.encode("utf-8"))
    offset = 0
    if args.hnet_bos:
        ids = [args.hnet_bos_id] + ids
        offset = 1
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(input_ids)

    masks = [bp.boundary_mask[0].bool().cpu().tolist() for bp in output.bpred_output]
    starts = compose_stage_starts(masks, len(ids), offset)

    del model, output
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    _STARTS_CACHE[cache_key] = starts
    return starts


def chunks_to_result(text: str, byte_starts, vocab: int = 256):
    """Posições de byte onde começa cada chunk -> (pieces, raw, ids, vocab).

    Uma fronteira que cai no meio de um caractere multibyte (ex.: 'ã' = 2 bytes) é
    movida para o início desse caractere, para as peças continuarem sendo texto válido.
    """
    b2c = []
    for ci, ch in enumerate(text):
        b2c.extend([ci] * len(ch.encode("utf-8")))
    cuts = sorted({0} | {b2c[b] for b in byte_starts if 0 <= b < len(b2c)})
    edges = cuts + [len(text)]
    pieces = [text[a:b] for a, b in zip(edges[:-1], edges[1:])]
    raw = [" ".join(f"{x:02x}" for x in p.encode("utf-8")) for p in pieces]
    return pieces, raw, None, vocab


def _make_tok_hnet(key: str, stage: int | None):
    """stage=None -> último estágio (chunks mais grossos); inteiro -> aquele estágio."""
    def fn(text, args):
        per_stage = hnet_stage_starts(key, text, args)
        s = len(per_stage) - 1 if stage is None else stage
        return chunks_to_result(text, per_stage[s])
    return fn


TOKENIZERS = {
    "bpe": ("BPE byte-level (GPT-2)", tok_bpe),
    "tiktoken": ("BPE byte-level (tiktoken)", tok_tiktoken),
    "wordpiece": ("WordPiece (BERT)", tok_wordpiece),
    "sentencepiece": ("SentencePiece Unigram (XLM-R)", tok_sentencepiece),
    "spacy": ("spaCy (regras)", tok_spacy),
    "bytes": ("Bytes UTF-8", tok_bytes),
}
for _k, _label in HNET_MODELS.items():
    TOKENIZERS[_k] = (_label, _make_tok_hnet(_k, None))
TOKENIZERS["hnet_2stage_L_s0"] = ("H-Net 2-stage (L) — estágio 0 (mais fino)",
                                  _make_tok_hnet("hnet_2stage_L", 0))


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

    p.add_argument("--tokenizers", default="bpe,tiktoken,wordpiece,sentencepiece,spacy,bytes",
                   help=f"lista separada por vírgula; válidos: {','.join(TOKENIZERS)}")
    p.add_argument("--bpe-name", default="gpt2")
    p.add_argument("--tiktoken-name", default="cl100k_base")
    p.add_argument("--wordpiece-name", default="bert-base-multilingual-cased")
    p.add_argument("--sp-name", default="xlm-roberta-base")
    p.add_argument("--spacy-model", default="pt_core_news_sm")
    p.add_argument("--spacy-lang", default="pt", help="idioma do spacy.blank (fallback)")

    # H-Net: config (json do build_model) + checkpoint de cada modelo
    for key in HNET_MODELS:
        flag = key.replace("_", "-")
        p.add_argument(f"--{flag}-config", dest=f"{key}_config", default=None,
                       help=f"json de configuração do modelo {key} (para build_model)")
        p.add_argument(f"--{flag}-ckpt", dest=f"{key}_ckpt", default=None,
                       help=f"caminho do checkpoint do modelo {key}")
    p.add_argument("--hnet-repo", default=None,
                   help="diretório do repositório H-Net, adicionado ao sys.path")
    p.add_argument("--hnet-device", default=None, help="padrão: cuda se disponível, senão cpu")
    p.add_argument("--hnet-dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--hnet-bos", action=argparse.BooleanOptionalAction, default=True,
                   help="prefixa o token BOS aos bytes (convenção do H-Net original)")
    p.add_argument("--hnet-bos-id", type=int, default=254)

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
        print(f"{'tokenizador':<20}{'tokens':>8}{'chars/tok':>12}{'bytes/tok':>12}")
        for r in sorted(results, key=lambda r: r["n_tokens"]):
            print(f"{r['tokenizer']:<20}{r['n_tokens']:>8}"
                  f"{r['chars_per_token']:>12.2f}{r['bytes_per_token']:>12.2f}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "text": text, "results": results}, f,
                      ensure_ascii=False, indent=2)
        print(f"\nSalvo em {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())