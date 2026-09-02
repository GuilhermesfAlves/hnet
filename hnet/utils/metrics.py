import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class HNetMetrics:
    """Acumula métricas ao longo do treinamento."""
    
    # BPB
    total_nll_bits: float = 0.0
    total_bytes: int = 0
    
    # Compression ratio por estágio
    compression_ratios: dict = field(default_factory=dict)
    
    # Ratio loss
    ratio_losses: list = field(default_factory=list)
    
    # Steps acumulados
    steps: int = 0


# ─────────────────────────────────────────────
# 1. BITS-PER-BYTE (BPB)
# ─────────────────────────────────────────────

def compute_bpb(
    logits: torch.Tensor,       # (B, L, vocab_size)
    targets: torch.Tensor,      # (B, L) — bytes 0-255
    bytes_per_token: float = 1.0,  # 1.0 para byte-level; ~4.6 para BPE+GPT2
) -> torch.Tensor:
    """
    BPB = NLL_em_nats / ln(2) / bytes_por_token
    
    Para modelos byte-level: bytes_per_token = 1.0
    Para BPE: bytes_per_token = média de bytes por token do dataset
              (ex: 4.6 para GPT-2 no FineWeb-Edu)
    """
    B, L, V = logits.shape
    
    nll_nats = F.cross_entropy(
        logits.reshape(B * L, V),
        targets.reshape(B * L),
        reduction="mean",
    )
    
    # Conversão: nats → bits, ajuste por token
    bpb = nll_nats / math.log(2) / bytes_per_token
    return bpb


def bpb_from_total_nll(
    total_nll_nats: float,
    total_tokens: int,
    bytes_per_token: float = 1.0,
) -> float:
    """Versão acumulada para eval sobre dataset inteiro."""
    avg_nll = total_nll_nats / total_tokens
    return avg_nll / math.log(2) / bytes_per_token


# ─────────────────────────────────────────────
# 2. COMPRESSION RATIO (Lˢ⁺¹ / Lˢ)
# ─────────────────────────────────────────────

def compute_compression_ratio(
    boundary_indicators: torch.Tensor,  # (B, L) — valores binários b_t ∈ {0, 1}
) -> torch.Tensor:
    """
    Razão de compressão = fração de posições marcadas como boundary.
    
    boundary_indicators[b, t] = 1 → posição t é início de chunk
                               = 0 → posição descartada
    
    Retorna: razão média no batch (escalar)
    """
    # Fração de 1s por sequência, média no batch
    ratio = boundary_indicators.float().mean()
    return ratio


def compute_stage_ratios(
    boundary_indicators_per_stage: list[torch.Tensor],
) -> dict[str, float]:
    """
    Para modelos multi-stage (ex: 2-stage com L0→L1→L2).
    
    boundary_indicators_per_stage[s] tem shape (B, Lˢ)
    
    Retorna dict: {"L1/L0": 0.33, "L2/L1": 0.34, ...}
    """
    ratios = {}
    for s, b_indicators in enumerate(boundary_indicators_per_stage):
        key = f"L{s+1}/L{s}"
        ratios[key] = compute_compression_ratio(b_indicators).item()
    return ratios


# ─────────────────────────────────────────────
# 3. RATIO LOSS (Equação 10 do paper)
# ─────────────────────────────────────────────

def compute_ratio_loss(
    boundary_probs: torch.Tensor,       # (B, L) — p_t ∈ [0, 1], contínuo
    boundary_indicators: torch.Tensor,  # (B, L) — b_t ∈ {0, 1}, discreto
    N: float = 6.0,                     # fator de compressão alvo (ex: 6 → 1/6 dos tokens)
) -> torch.Tensor:
    """
    Lratio = N/(N-1) * ((N-1)*F*G + (1-F)*(1-G))
    
    F = fração real de boundaries selecionados (não diferenciável)
    G = média das probabilidades (diferenciável — usado no backprop)
    
    Mínimo em F = G = 1/N → razão alvo
    """
    # F: fração de boundaries reais (stop gradient, serve só de sinal)
    F = boundary_indicators.float().mean()
    
    # G: média das probabilidades (diferenciável)
    G = boundary_probs.mean()
    
    ratio_loss = (N / (N - 1)) * ((N - 1) * F * G + (1 - F) * (1 - G))
    return ratio_loss


def compute_total_loss(
    ar_loss: torch.Tensor,
    ratio_losses_per_stage: list[torch.Tensor],
    alpha: float = 0.03,            # peso fixo usado no paper
) -> tuple[torch.Tensor, dict]:
    """
    L = L_AR + α * Σ_s L_ratio^s
    
    alpha = 0.03 fixo em todos os experimentos do paper
    """
    ratio_total = sum(ratio_losses_per_stage)
    total = ar_loss + alpha * ratio_total
    
    log_dict = {
        "loss/ar": ar_loss.item(),
        "loss/ratio": ratio_total.item(),
        "loss/total": total.item(),
    }
    return total, log_dict


# ─────────────────────────────────────────────
# 4. BYTES-PER-INNERMOST-CHUNK (BPIC)
# ─────────────────────────────────────────────

def compute_bpic(
    L0: int,    # comprimento da sequência no estágio mais externo (bytes brutos)
    LS: int,    # comprimento no estágio mais interno (innermost chunks)
) -> float:
    """
    BPIC = L0 / LS
    
    Mede quantos bytes brutos correspondem a cada chunk no estágio mais interno.
    No paper: BPIC ≈ 4.5–5 para 1-stage (similar ao GPT-2 tokenizer com ~4.6 bytes/token)
    """
    return L0 / LS


def compute_bpic_from_boundaries(
    boundary_indicators_per_stage: list[torch.Tensor],
    L0: int,
) -> float:
    """
    Versão que calcula BPIC a partir dos boundary indicators de cada estágio.
    
    Para S estágios: BPIC = L0 / (L0 * ratio_0 * ratio_1 * ... * ratio_{S-1})
    """
    compound_ratio = 1.0
    for b_ind in boundary_indicators_per_stage:
        compound_ratio *= b_ind.float().mean().item()
    
    LS_estimated = L0 * compound_ratio
    return L0 / LS_estimated if LS_estimated > 0 else float("inf")


# ─────────────────────────────────────────────
# 5. ROBUSTNESS SCORE (Apêndice D.1)
# ─────────────────────────────────────────────

def compute_robustness_score(
    perturbed_acc: float,
    unperturbed_acc: float,
    chance_level: float = 0.25,   # HellaSwag tem 4 opções → chance = 0.25
) -> float:
    """
    robustness_score = 100 * (perturbed_acc - chance) / max(unperturbed_acc - chance, 0)
    
    Mede % da performance original mantida sob perturbação.
    Score = 100 → nenhuma degradação
    Score = 0   → performance caiu ao nível do acaso
    """
    numerator = perturbed_acc - chance_level
    denominator = max(unperturbed_acc - chance_level, 1e-8)
    return 100.0 * numerator / denominator


# ─────────────────────────────────────────────
# 6. LOOP DE TREINAMENTO — INTEGRAÇÃO COMPLETA
# ─────────────────────────────────────────────

def training_step(
    model,
    batch: dict,
    optimizer,
    step: int,
    N_stages: list[float],   # ex: [6.0] para 1-stage ou [3.0, 3.0] para 2-stage
    alpha: float = 0.03,
    bytes_per_gpt2_token: float = 4.6,
) -> dict:
    """
    Exemplo de step de treinamento com todas as métricas do paper.
    
    Assume que model retorna:
        logits                  — (B, L0, vocab)
        boundary_probs_list     — [(B, L0), (B, L1), ...]  uma por estágio
        boundary_ind_list       — [(B, L0), (B, L1), ...]  versão discreta
    """
    model.train()
    optimizer.zero_grad()

    input_bytes = batch["input_ids"]   # (B, L0)
    target_bytes = batch["labels"]     # (B, L0)

    # Forward
    logits, boundary_probs_list, boundary_ind_list = model(input_bytes)

    # ── Loss principal ──────────────────────────────────
    B, L, V = logits.shape
    ar_loss = F.cross_entropy(logits.reshape(B * L, V), target_bytes.reshape(B * L))

    # ── Ratio losses por estágio ────────────────────────
    ratio_losses = []
    for s, (b_probs, b_inds, N) in enumerate(
        zip(boundary_probs_list, boundary_ind_list, N_stages)
    ):
        rl = compute_ratio_loss(b_probs, b_inds, N=N)
        ratio_losses.append(rl)

    total_loss, loss_log = compute_total_loss(ar_loss, ratio_losses, alpha)

    # ── Backward ────────────────────────────────────────
    total_loss.backward()
    optimizer.step()

    # ── Métricas de monitoramento ───────────────────────
    with torch.no_grad():
        bpb = compute_bpb(logits, target_bytes, bytes_per_token=1.0)
        bpb_vs_bpe = compute_bpb(logits, target_bytes, bytes_per_token=bytes_per_gpt2_token)
        stage_ratios = compute_stage_ratios(boundary_ind_list)
        bpic = compute_bpic_from_boundaries(boundary_ind_list, L0=input_bytes.shape[1])

    metrics = {
        **loss_log,
        "metrics/bpb": bpb.item(),
        "metrics/bpb_vs_bpe_scale": bpb_vs_bpe.item(),
        "metrics/bpic": bpic,
        **{f"metrics/compression_{k}": v for k, v in stage_ratios.items()},
        "step": step,
    }

    return metrics


# ─────────────────────────────────────────────
# 7. LOOP DE AVALIAÇÃO
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, eval_loader, device, bytes_per_token: float = 1.0) -> dict:
    """Avaliação completa sobre o dataset de validação."""
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    all_ratios = {f"L{s+1}/L{s}": [] for s in range(model.num_stages)}
    all_bpics = []

    for batch in eval_loader:
        input_bytes = batch["input_ids"].to(device)
        target_bytes = batch["labels"].to(device)
        B, L = input_bytes.shape

        logits, _, boundary_ind_list = model(input_bytes)

        nll = F.cross_entropy(
            logits.reshape(B * L, -1),
            target_bytes.reshape(B * L),
            reduction="sum",
        )
        total_nll += nll.item()
        total_tokens += B * L

        for s, b_ind in enumerate(boundary_ind_list):
            all_ratios[f"L{s+1}/L{s}"].append(
                compute_compression_ratio(b_ind).item()
            )
        all_bpics.append(compute_bpic_from_boundaries(boundary_ind_list, L0=L))

    # Agregação
    avg_nll = total_nll / total_tokens
    bpb = avg_nll / math.log(2) / bytes_per_token

    return {
        "eval/bpb": bpb,
        "eval/bpic": sum(all_bpics) / len(all_bpics),
        **{
            f"eval/compression_{k}": sum(v) / len(v)
            for k, v in all_ratios.items()
        },
    }