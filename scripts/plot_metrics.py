"""
plot_metrics.py

Lê o CSV gerado pelo CsvLogger de train_pt.py e gera gráficos
das métricas de treino e validação, quando disponíveis.

Uso:
    python plot_metrics.py \
        --csv checkpoints/pt-hnet/metrics.csv \
        --out-dir plots/
"""

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_number(value):
    """Converte um valor CSV para float ou retorna None se inválido."""
    if value is None or not value.strip():
        return None

    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def load_metrics(csv_path: str):
    """
    Carrega as linhas do CSV, separando treino e validação.

    Colunas numéricas são convertidas individualmente, permitindo
    que métricas ausentes não invalidem uma linha inteira.
    """
    train_rows = []
    val_rows = []

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            return train_rows, val_rows, []

        fields = reader.fieldnames

        for row in reader:
            step = parse_number(row.get("step"))
            if step is None:
                continue

            row["step"] = step

            # Converte individualmente todas as colunas numéricas.
            for field in fields:
                if field not in ("split", "step"):
                    row[field] = parse_number(row.get(field))

            split = (row.get("split") or "").strip().lower()

            if split == "train":
                train_rows.append(row)
            elif split in ("val", "valid", "validation"):
                val_rows.append(row)

    return train_rows, val_rows, fields


def metric_label(key: str) -> str:
    """Gera um rótulo legível para cada métrica."""
    labels = {
        "lm_loss": "Loss (LM, cross-entropy)",
        "perplexity": "Perplexity",
        "lb_loss": "Load balancing loss",
        "ratio_loss": "Ratio loss",
        "total_loss": "Loss total",
        "bpb": "Bits per byte (BPB)",
        "bpic": "Bytes per innermost chunk (BPIC)",
        "lr": "Learning rate",
        "tokens_per_sec": "Tokens por segundo",
        "wall_time": "Tempo decorrido (s)",
        "tokens": "Tokens processados",
    }

    if key in labels:
        return labels[key]

    # Ex.: compression_L1/L0 -> Compression L1/L0
    return key.replace("_", " ").replace("/", " / ")


def safe_filename(name: str) -> str:
    """Converte o nome da coluna em um nome de arquivo seguro."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")


def plot_metric(
    train_rows,
    val_rows,
    key: str,
    out_path: Path,
    log_scale: bool = False,
):
    """Gera um gráfico de uma métrica para treino e validação."""
    def valid_points(rows):
        points = []

        for row in rows:
            value = row.get(key)

            if value is not None and math.isfinite(value):
                if not log_scale or value > 0:
                    points.append((row["step"], value))

        return points

    train_points = valid_points(train_rows)
    val_points = valid_points(val_rows)

    if not train_points and not val_points:
        return False

    fig, ax = plt.subplots(figsize=(9, 5))

    if train_points:
        steps, values = zip(*train_points)
        ax.plot(
            steps,
            values,
            label="Treino",
            alpha=0.85,
        )

    if val_points:
        steps, values = zip(*val_points)
        ax.plot(
            steps,
            values,
            label="Validação",
            marker="o",
            linestyle="--",
        )

    label = metric_label(key)

    ax.set_xlabel("Passo de treinamento")
    ax.set_ylabel(label)
    ax.set_title(label)

    if log_scale:
        ax.set_yscale("log")

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Salvo: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Plota métricas de treinamento do H-Net."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Caminho do metrics.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="plots",
        help="Diretório de saída dos gráficos",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)

    if not csv_path.is_file():
        parser.error(f"Arquivo CSV não encontrado: {csv_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows, fields = load_metrics(str(csv_path))

    if not train_rows and not val_rows:
        print("Nenhuma linha válida de treino ou validação encontrada.")
        return

    # Métricas principais, na ordem desejada.
    preferred_metrics = [
        "lm_loss",
        "perplexity",
        "lb_loss",
        "ratio_loss",
        "total_loss",
        "bpb",
        "bpic",
        "lr",
        "tokens_per_sec",
        "wall_time",
        "tokens",
    ]

    # Inclui automaticamente todas as métricas de compressão
    # presentes no cabeçalho, independentemente do número de estágios.
    compression_metrics = [
        field
        for field in fields
        if field.startswith("compression_")
    ]

    metrics = [
        key
        for key in preferred_metrics + compression_metrics
        if key in fields
    ]

    # Perplexity é positiva e pode variar em várias ordens de grandeza.
    log_metrics = {"perplexity"}

    generated = 0

    for key in metrics:
        out_path = out_dir / f"{safe_filename(key)}.png"

        if plot_metric(
            train_rows,
            val_rows,
            key,
            out_path,
            log_scale=key in log_metrics,
        ):
            generated += 1

    if generated == 0:
        print("Nenhuma métrica numérica válida encontrada para plotar.")
    else:
        print(f"\nConcluído: {generated} gráfico(s) em {out_dir}")


if __name__ == "__main__":
    main()
