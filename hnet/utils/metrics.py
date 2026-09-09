"""
hnet/utils/metrics.py

Métricas do paper H-Net (bits-per-byte, taxa de compressão, ratio loss, BPIC)
e utilitários de agregação para treino/avaliação.

Assinaturas alinhadas com o uso em train_pt.py:
    compute_bpb(avg_nll_nats: float, bytes_per_token: float = 1.0) -> float
    compute_bpic(L0: int, boundary_ind_list: list[Tensor]) -> float
    compute_compression_ratio(boundary_indicators: Tensor) -> float
    compute_ratio_loss(boundary_probs, boundary_indicators, N) -> Tensor  (diferenciável)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist


# --------------------------------------------------------------------------- #
# 1. Bits-per-byte (BPB)
# --------------------------------------------------------------------------- #
def compute_bpb(avg_nll_nats: float, bytes_per_token: float = 1.0) -> float:
    """
    BPB = NLL_em_nats / ln(2) / bytes_por_token

    `avg_nll_nats` é a cross-entropy média (já em nats), ex: lm_loss.item().
    Para modelos byte-level: bytes_per_token = 1.0.
    Para comparar na escala de um tokenizer BPE: bytes_per_token = média de
    bytes por token do dataset (ex: ~4.6 para GPT-2 / FineWeb-Edu).
    """
    return avg_nll_nats / math.log(2) / bytes_per_token


def bpb_from_total_nll(total_nll_nats: float, total_tokens: int, bytes_per_token: float = 1.0) -> float:
    """Versão acumulada (soma exata) para avaliação sobre um dataset inteiro."""
    if total_tokens == 0:
        return float("nan")
    avg_nll = total_nll_nats / total_tokens
    return compute_bpb(avg_nll, bytes_per_token)


# --------------------------------------------------------------------------- #
# 2. Compression ratio (Lˢ⁺¹ / Lˢ)
# --------------------------------------------------------------------------- #
def compute_compression_ratio(boundary_indicators: torch.Tensor) -> float:
    """
    Razão de compressão de um estágio = fração de posições marcadas como
    boundary (b_t = 1), média no batch inteiro. Retorna float puro (não
    tensor), pois é usada apenas para logging/monitoramento.
    """
    return boundary_indicators.float().mean().item()


def compute_stage_ratios(boundary_indicators_per_stage: list) -> dict:
    """
    Para modelos multi-estágio (ex: 2-stage com L0→L1→L2).
    Retorna dict {"L1/L0": ratio, "L2/L1": ratio, ...}
    """
    ratios = {}
    for s, b_indicators in enumerate(boundary_indicators_per_stage):
        key = f"L{s + 1}/L{s}"
        ratios[key] = compute_compression_ratio(b_indicators)
    return ratios


# --------------------------------------------------------------------------- #
# 3. Ratio loss (Equação 10 do paper) — DIFERENCIÁVEL
# --------------------------------------------------------------------------- #
def compute_ratio_loss(
    boundary_probs: torch.Tensor,       # (B, L) — p_t ∈ [0, 1], contínuo
    boundary_indicators: torch.Tensor,  # (B, L) — b_t ∈ {0, 1}, discreto
    N: float = 6.0,                     # fator de compressão alvo (1/N)
) -> torch.Tensor:
    """
    Lratio = N/(N-1) * ((N-1)*F*G + (1-F)*(1-G))

    F = fração real de boundaries selecionados (constante — só guia o sinal)
    G = média das probabilidades (diferenciável — recebe o gradiente)

    Mínimo global em F = G = 1/N → taxa de compressão alvo.
    Retorna um tensor (mantém o grafo computacional, necessário para o
    backward em train_pt.py).
    """
    F_val = boundary_indicators.float().mean()   # sem gradiente (é 0/1, detached na prática)
    G_val = boundary_probs.mean()

    return (N / (N - 1)) * ((N - 1) * F_val * G_val + (1 - F_val) * (1 - G_val))


def compute_total_loss(
    ar_loss: torch.Tensor,
    ratio_losses_per_stage: list,
    alpha: float = 0.03,
) -> tuple:
    """L = L_AR + α * Σ_s L_ratio^s  (alpha = 0.03 fixo no paper)"""
    ratio_total = (
        sum(ratio_losses_per_stage)
        if ratio_losses_per_stage
        else torch.zeros((), device=ar_loss.device)
    )
    total = ar_loss + alpha * ratio_total

    log_dict = {
        "loss/ar": ar_loss.item(),
        "loss/ratio": ratio_total.item() if torch.is_tensor(ratio_total) else float(ratio_total),
        "loss/total": total.item(),
    }
    return total, log_dict


# --------------------------------------------------------------------------- #
# 4. Bytes-per-innermost-chunk (BPIC)
# --------------------------------------------------------------------------- #
def compute_bpic(L0: int, boundary_ind_list: list) -> float:
    """
    BPIC = L0 / LS, estimado compondo as razões de compressão de cada estágio:

        LS ≈ L0 * ratio_0 * ratio_1 * ... * ratio_{S-1}

    No paper: BPIC ≈ 4.5–5 para 1-stage (parecido com o tokenizer GPT-2, que
    tem ~4.6 bytes/token).

    `boundary_ind_list` é a lista de tensores boundary_ind por estágio,
    exatamente como vem em output.boundary_ind_list.
    """
    compound_ratio = 1.0
    for b_ind in boundary_ind_list:
        compound_ratio *= b_ind.float().mean().item()

    LS_estimated = L0 * compound_ratio
    return L0 / LS_estimated if LS_estimated > 0 else float("inf")


# --------------------------------------------------------------------------- #
# 5. Robustness score (Apêndice D.1)
# --------------------------------------------------------------------------- #
def compute_robustness_score(
    perturbed_acc: float,
    unperturbed_acc: float,
    chance_level: float = 0.25,   # HellaSwag: 4 opções → chance = 0.25
) -> float:
    """
    robustness_score = 100 * (perturbed_acc - chance) / max(unperturbed_acc - chance, eps)
    100 → nenhuma degradação. 0 → performance caiu ao nível do acaso.
    """
    numerator = perturbed_acc - chance_level
    denominator = max(unperturbed_acc - chance_level, 1e-8)
    return 100.0 * numerator / denominator


# --------------------------------------------------------------------------- #
# 6. Acumulador de métricas — usado no loop de treino E na validação
# --------------------------------------------------------------------------- #
@dataclass
class HNetMetrics:
    """
    Acumula métricas ao longo de uma janela (intervalo de log no treino, ou
    uma passada inteira de validação) e depois produz médias/valores exatos
    via `averages()`.

    - `total_nll_nats` / `total_tokens`: somas exatas -> BPB e lm_loss exatos
      (não é média de médias, é NLL total / tokens totais).
    - `sum_lb_loss` / `sum_ratio_loss` / `sum_bpic` / `stage_ratio_sums`:
      somas por batch/microbatch -> divididas por `n_updates` -> médias.

    Uso típico no treino:
        train_metrics = HNetMetrics()
        ...
        train_metrics.update(
            nll_sum_nats=lm_loss.item() * B * L0,
            n_tokens=B * L0,
            lb_loss=lb_loss.item(),
            ratio_loss=ratio_loss.item(),
            bpic=bpic,
            stage_ratios=stage_ratios,
        )
        ...
        if is_distributed:
            train_metrics.all_reduce_(device)
        avgs = train_metrics.averages()
        train_metrics.reset()
    """

    total_nll_nats: float = 0.0     # soma exata de NLL (nats) sobre todos os tokens
    total_tokens: int = 0           # soma exata de tokens correspondentes

    sum_lb_loss: float = 0.0
    sum_ratio_loss: float = 0.0
    sum_bpic: float = 0.0
    stage_ratio_sums: dict = field(default_factory=dict)

    n_updates: int = 0              # nº de chamadas a update() (batches/microbatches)

    def update(
        self,
        nll_sum_nats: float,
        n_tokens: int,
        lb_loss: float = 0.0,
        ratio_loss: float = 0.0,
        bpic: float = 0.0,
        stage_ratios: Optional[dict] = None,
    ) -> None:
        self.total_nll_nats += nll_sum_nats
        self.total_tokens += n_tokens
        self.sum_lb_loss += lb_loss
        self.sum_ratio_loss += ratio_loss
        self.sum_bpic += bpic
        if stage_ratios:
            for k, v in stage_ratios.items():
                self.stage_ratio_sums[k] = self.stage_ratio_sums.get(k, 0.0) + v
        self.n_updates += 1

    def reset(self) -> None:
        self.total_nll_nats = 0.0
        self.total_tokens = 0
        self.sum_lb_loss = 0.0
        self.sum_ratio_loss = 0.0
        self.sum_bpic = 0.0
        self.stage_ratio_sums = {}
        self.n_updates = 0

    def all_reduce_(self, device) -> None:
        """
        Soma os acumuladores entre todos os ranks (DDP), in-place. Precisa
        ser chamado por TODOS os ranks (é uma operação coletiva) e sempre
        antes de `averages()`/`reset()` no rank principal.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return
        keys = sorted(self.stage_ratio_sums.keys())
        payload = torch.tensor(
            [
                self.total_nll_nats,
                float(self.total_tokens),
                self.sum_lb_loss,
                self.sum_ratio_loss,
                self.sum_bpic,
                float(self.n_updates),
            ]
            + [self.stage_ratio_sums[k] for k in keys],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        vals = payload.tolist()

        self.total_nll_nats = vals[0]
        self.total_tokens = int(vals[1])
        self.sum_lb_loss = vals[2]
        self.sum_ratio_loss = vals[3]
        self.sum_bpic = vals[4]
        self.n_updates = int(vals[5])
        for k, v in zip(keys, vals[6:]):
            self.stage_ratio_sums[k] = v

    def averages(self, bytes_per_token: float = 1.0) -> dict:
        """
        Retorna {} se nada foi acumulado ainda. Caso contrário retorna:
            lm_loss, perplexity, bpb  -> exatos (soma total / tokens totais)
            lb_loss, ratio_loss, bpic -> médias por batch/microbatch
            stage_ratios              -> dict de médias por estágio
        """
        if self.total_tokens == 0 or self.n_updates == 0:
            return {}

        avg_lm = self.total_nll_nats / self.total_tokens
        bpb = compute_bpb(avg_lm, bytes_per_token)

        return {
            "lm_loss": avg_lm,
            "perplexity": math.exp(min(avg_lm, 20)),
            "bpb": bpb,
            "lb_loss": self.sum_lb_loss / self.n_updates,
            "ratio_loss": self.sum_ratio_loss / self.n_updates,
            "bpic": self.sum_bpic / self.n_updates,
            "stage_ratios": {k: v / self.n_updates for k, v in self.stage_ratio_sums.items()},
        }