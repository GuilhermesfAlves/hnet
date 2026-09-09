from datasets import load_dataset
import numpy as np
from hnet.utils import ByteTokenizer

# ============================================================
# Configuração
# ============================================================

DATASET_NAME = "uonlp/CulturaX"
CONFIG_NAME = "pt"
SPLIT = "train"

# Coluna de texto do CulturaX
TEXT_COLUMN = "text"

# ============================================================
# Carregar tokenizer
# ============================================================

tokenizer = ByteTokenizer() 

# ============================================================
# Carregar dataset em streaming
# ============================================================

dataset = load_dataset(
    DATASET_NAME,
    CONFIG_NAME,
    split=SPLIT,
    streaming=True,
)

# ============================================================
# Contagem
# ============================================================

total_entries = 0
total_tokens = 0

min_tokens = None
max_tokens = 0

lengths = []

for example in dataset:
    text = example[TEXT_COLUMN]

    if not text:
        continue

    # Não adiciona tokens especiais (BOS/EOS etc.)
    n_tokens = len(
        tokenizer.encode(
            text
        )
    )

    total_entries += 1
    total_tokens += n_tokens

    min_tokens = n_tokens if min_tokens is None else min(min_tokens, n_tokens)
    max_tokens = max(max_tokens, n_tokens)

    lengths.append(n_tokens)

    if total_entries % 100_000 == 0:
        print(
            f"Processados: {total_entries:,} | "
            f"Tokens: {total_tokens:,} | "
            f"Média: {total_tokens / total_entries:,.2f}"
        )

# ============================================================
# Estatísticas
# ============================================================

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
print(f"Mínimo:               {min_tokens:,}")
print(f"Máximo:               {max_tokens:,}")

# Percentis
lengths = np.array(lengths)

print(f"Mediana (P50):        {np.percentile(lengths, 50):,.2f}")
print(f"P90:                  {np.percentile(lengths, 90):,.2f}")
print(f"P95:                  {np.percentile(lengths, 95):,.2f}")
print(f"P99:                  {np.percentile(lengths, 99):,.2f}")
