import json, os
from datasets import load_dataset

OUT = "benchmarks_text"

def dump(name, rows):
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "train.jsonl"), "w", encoding="utf-8") as f:
        for t in rows:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    print(name, "ok")

# CALAME-PT (GlórIA) — VERIFICAR id/colunas
ds = load_dataset("NOVA-vision-language/calame-pt", split="train")
dump("calame_pt", (f'{r["sentence"]} {r["last_word"]}' for r in ds))

# ARC-Challenge-PT (Okapi) — VERIFICAR id/colunas
ds = load_dataset("lighteval/okapi_arc_challenge", "pt", split="test")
def arc(r):
    i = r["choices"]["label"].index(r["answerKey"])
    return f'{r["question"]} {r["choices"]["text"][i]}'
dump("arc_challenge_pt", (arc(r) for r in ds))

# HellaSwag-PT (Okapi) — VERIFICAR id/colunas
ds = load_dataset("lighteval/okapi_hellaswag", "pt", split="validation")
dump("hellaswag_pt", (f'{r["ctx"]} {r["endings"][int(r["label"])]}' for r in ds))

# LAMBADA-PT (usado pelo Tucano) — VERIFICAR id
ds = load_dataset("TucanoBR/lambada-pt", split="test")
dump("lambada_pt", (r["text"] for r in ds))