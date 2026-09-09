import os
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# Token da API do Hugging Face
HF_TOKEN = os.environ["HUGGING_FACE_TOKEN"]

# Dataset no formato "usuario/dataset"
DATASET = "uonlp/CulturaX"

load_dataset(
    DATASET,
    "pt",
    token=HF_TOKEN
)

print(f"Dataset baixado")
