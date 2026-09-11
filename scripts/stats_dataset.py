import time
import multiprocessing as mp
from functools import partial

import numpy as np
from datasets import load_dataset

DATASET_NAME = "uonlp/CulturaX"
CONFIG_NAME = "pt"
SPLIT = "train"
TEXT_COLUMN = "text"
BATCH_SIZE = 4096
NUM_WORKERS = 16


def process_shard(shard_id, dataset, num_shards, text_column, batch_size):
    try:
        shard = dataset.shard(num_shards=num_shards, index=shard_id, contiguous=True)

        total_entries = 0
        total_tokens = 0
        min_tokens = None
        max_tokens = None
        token_counts = []

        for start in range(0, len(shard), batch_size):
            batch = shard[start:start + batch_size]
            texts = batch[text_column]
            counts = [len(t.encode("utf-8")) for t in texts if t]
            if not counts:
                continue

            total_entries += len(counts)
            total_tokens += sum(counts)
            mn, mx = min(counts), max(counts)
            min_tokens = mn if min_tokens is None else min(min_tokens, mn)
            max_tokens = mx if max_tokens is None else max(max_tokens, mx)
            token_counts.extend(counts)

        return {
            "ok": True,
            "shard_id": shard_id,
            "total_entries": total_entries,
            "total_tokens": total_tokens,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "token_counts": token_counts,
        }
    except Exception as e:
        # não deixa uma falha de um shard matar o pool inteiro
        return {"ok": False, "shard_id": shard_id, "error": str(e)}


def main():
    t0 = time.time()

    print("Carregando dataset (uma única vez, no processo pai)...")
    dataset = load_dataset(DATASET_NAME, CONFIG_NAME, split=SPLIT, streaming=False)
    print(f"Dataset carregado: {len(dataset):,} entradas")

    print(f"Disparando {NUM_WORKERS} workers via fork...")
    worker_fn = partial(
        process_shard,
        dataset=dataset,          # herdado via fork, sem recarregar
        num_shards=NUM_WORKERS,
        text_column=TEXT_COLUMN,
        batch_size=BATCH_SIZE,
    )

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=NUM_WORKERS) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(worker_fn, range(NUM_WORKERS)), 1):
            if not r["ok"]:
                print(f"[{i}/{NUM_WORKERS}] Shard {r['shard_id']} FALHOU: {r['error']}")
                continue
            print(f"[{i}/{NUM_WORKERS}] Shard {r['shard_id']} OK: "
                  f"{r['total_entries']:,} entradas | {r['total_tokens']:,} tokens")
            results.append(r)

    if not results:
        print("Nenhum shard processado com sucesso. Abortando.")
        return

    total_entries = sum(r["total_entries"] for r in results)
    total_tokens = sum(r["total_tokens"] for r in results)
    min_tokens = min(r["min_tokens"] for r in results if r["min_tokens"] is not None)
    max_tokens = max(r["max_tokens"] for r in results if r["max_tokens"] is not None)

    all_token_counts = []
    for r in results:
        all_token_counts.extend(r["token_counts"])

    lengths = np.asarray(all_token_counts)
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("RESULTADO")
    print("=" * 60)
    print(f"Workers OK:           {len(results)}/{NUM_WORKERS}")
    print(f"Tempo total:          {elapsed:,.1f}s")
    print(f"Entradas:             {total_entries:,}")
    print(f"Tokens totais:        {total_tokens:,}")
    print(f"Média tokens/entrada: {total_tokens / total_entries:,.2f}")
    print(f"Mínimo:               {min_tokens:,}")
    print(f"Máximo:               {max_tokens:,}")
    print(f"Mediana (P50):        {np.percentile(lengths, 50):,.2f}")
    print(f"P90:                  {np.percentile(lengths, 90):,.2f}")
    print(f"P95:                  {np.percentile(lengths, 95):,.2f}")
    print(f"P99:                  {np.percentile(lengths, 99):,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()