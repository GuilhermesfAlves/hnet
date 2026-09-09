"""
plot_metrics.py

Lê o CSV gerado por `train_pt.py` (colunas: step, split, lm_loss, perplexity,
lb_loss, total_loss, lr, tokens_per_sec, wall_time) e gera gráficos de
loss e perplexity (treino vs. validação, quando houver).

Uso:

    python plot_metrics.py --csv checkpoints/pt-hnet/metrics.csv --out-dir plots/
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(csv_path: str):
    train_rows = []
    val_rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # ignora linhas de validação sem valor numérico, se houver
            try:
                row["step"] = int(row["step"])
                row["lm_loss"] = float(row["lm_loss"])
                row["perplexity"] = float(row["perplexity"])
                row["lb_loss"] = float(row["lb_loss"])
                row["total_loss"] = float(row["total_loss"])
            except (ValueError, KeyError):
                continue

            if row["split"] == "train":
                train_rows.append(row)
            elif row["split"] == "val":
                val_rows.append(row)

    return train_rows, val_rows


def plot_metric(train_rows, val_rows, key: str, ylabel: str, out_path: Path, log_scale: bool = False):
    fig, ax = plt.subplots(figsize=(8, 5))

    if train_rows:
        ax.plot(
            [r["step"] for r in train_rows],
            [r[key] for r in train_rows],
            label="treino",
            alpha=0.8,
        )
    if val_rows:
        ax.plot(
            [r["step"] for r in val_rows],
            [r[key] for r in val_rows],
            label="validação",
            marker="o",
            linestyle="--",
        )

    ax.set_xlabel("passo")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvo: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plota métricas de treino do H-Net")
    parser.add_argument("--csv", type=str, required=True, help="Caminho do metrics.csv")
    parser.add_argument("--out-dir", type=str, default="plots", help="Diretório de saída dos gráficos")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = load_metrics(args.csv)

    if not train_rows and not val_rows:
        print("Nenhuma linha válida encontrada no CSV.")
        return

    plot_metric(train_rows, val_rows, "lm_loss", "Loss (LM, cross-entropy)", out_dir / "loss.png")
    plot_metric(train_rows, val_rows, "perplexity", "Perplexity", out_dir / "perplexity.png", log_scale=True)
    plot_metric(train_rows, val_rows, "lb_loss", "Load balancing loss", out_dir / "load_balancing_loss.png")
    plot_metric(train_rows, val_rows, "total_loss", "Loss total (LM + balanceamento)", out_dir / "total_loss.png")


if __name__ == "__main__":
    main()
