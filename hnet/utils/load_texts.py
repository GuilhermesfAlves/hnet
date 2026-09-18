from pathlib import Path

def load_texts(path: str = "texts/input") -> list[str]:
    texts = []

    for i in range(1, 5):
        path_file = Path(path) / f"input_{i}.txt"

        with open(path_file, "r", encoding="utf-8") as f:
            texts.append(f.read())

    return texts