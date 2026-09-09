from datasets import load_dataset
import numpy as np
from hnet.utils import ByteTokenizer
from collections import deque

# ============================================================
# Configuração
# ============================================================

DATASET_NAME = "uonlp/CulturaX"
CONFIG_NAME = "pt"
SPLIT = "train"
TEXT_COLUMN = "text"
BATCH_SIZE = 1024  # Processar em batches

# ============================================================
# Carregar tokenizer
# ============================================================

tokenizer = ByteTokenizer() 

# ============================================================
# Carregar dataset (SEM streaming, com cache)
# ============================================================

print("Carregando dataset...")
dataset = load_dataset(
    DATASET_NAME,
    CONFIG_NAME,
    split=SPLIT,
    streaming=False,  
)

print(f"Dataset carregado: {len(dataset)} entradas")

# ============================================================
# Processar em batches com map()
# ============================================================

def tokenize_batch(batch):
    """Tokeniza um batch de textos"""
    texts = batch[TEXT_COLUMN]
    
    # Filtrar textos vazios
    filtered_texts = [t for t in texts if t]
    
    if not filtered_texts:
        return {"token_counts": []}
    
    # Tokenizar em batch (mais eficiente)
    token_counts = [len(tokenizer.encode(text)) for text in filtered_texts]
    
    return {"token_counts": token_counts}

print("Tokenizando dataset...")
dataset_tokens = dataset.map(
    tokenize_batch,
    batched=True,
    batch_size=BATCH_SIZE,
    remove_columns=[TEXT_COLUMN],  
    num_proc=8,
    desc="Tokenizando",
)

# ============================================================
# Extrai contagens e filtra vazios
# ============================================================

all_token_counts = []
total_entries = 0
total_tokens = 0

for item in dataset_tokens:
    counts = item["token_counts"]
    if counts:  # Pula se vazio
        all_token_counts.extend(counts)
        total_entries += len(counts)
        total_tokens += sum(counts)
    
    if total_entries % 500_000 == 0:
        print(
            f"Processados: {total_entries:,} | "
            f"Tokens: {total_tokens:,} | "
            f"Média: {total_tokens / total_entries:,.2f}"
        )

# ============================================================
# Estatísticas (com numpy array)
# ============================================================

lengths = np.array(all_token_counts)

mean_tokens = total_tokens / total_entries

print("\n" + "=" * 60)
print("RESULTADO")
print("=" * 60)

print(f"Dataset:              {DATASET_NAME}")
print(f"Config:               {CONFIG_NAME}")
print(f"Split:                {SPLIT}")
print(f"Entradas:             {total_entries:,}")
print(f"Tokens totais:        {total_tokens:,}")
print(f"Média tokens/entrada: {mean_tokens:,.2f}")
print(f"Mínimo:               {int(np.min(lengths)):,}")
print(f"Máximo:               {int(np.max(lengths)):,}")

print(f"Mediana (P50):        {np.percentile(lengths, 50):,.2f}")
print(f"P90:                  {np.percentile(lengths, 90):,.2f}")
print(f"P95:                  {np.percentile(lengths, 95):,.2f}")
print(f"P99:                  {np.percentile(lengths, 99):,.2f}")

print("=" * 60)