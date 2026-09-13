
import torch
import torch.nn.functional as F

from hnet.models.mixer_seq import find_frozen_external_modules
from hnet.utils.metrics import HNetMetrics, compute_bpic, compute_compression_ratio, compute_ratio_loss
from hnet.utils.train import load_balancing_loss


@torch.no_grad()
def evaluate(
    model,
    val_iter_factory,
    lb_n: list,
    device,
    eval_steps: int,
    bytes_per_token: float = 1.0,
):
    """
    Validação usando HNetMetrics para acumular:
      - soma exata de NLL/tokens  -> lm_loss e BPB exatos sobre todo o split
      - médias por batch          -> lb_loss, ratio_loss, bpic, stage_ratios

    Retorna: (lm_loss, lb_loss, ratio_loss, bpb, bpic, stage_ratios)
    """
    model.eval()
    data_iter   = val_iter_factory()
    val_metrics = HNetMetrics()

    # Manter backbones externos congelados em eval
    frozen_modules = find_frozen_external_modules(model)
    for m in frozen_modules:
        m.eval()

    for _ in range(eval_steps):
        try:
            input_ids, targets = next(data_iter)
        except StopIteration:
            break

        input_ids = input_ids.to(device)
        targets   = targets.to(device)
        B, L0     = input_ids.shape

        output = model(input_ids)
        logits = output.logits

        # NLL exata (reduction="sum") -> soma-se ao total para BPB exato
        nll_sum = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
            reduction="sum",
        )

        lb_loss = torch.zeros((), device=device)
        if hasattr(output, "bpred_output") and lb_n:
            for bpred, n in zip(output.bpred_output, lb_n):
                lb_loss = lb_loss + load_balancing_loss(bpred, n)

        ratio_loss   = torch.zeros((), device=device)
        stage_ratios = {}
        bpic         = 0.0
        has_hnet = (
            hasattr(output, "boundary_probs_list") and
            hasattr(output, "boundary_ind_list")
        )

        if has_hnet and lb_n:
            for b_probs, b_inds, N in zip(
                output.boundary_probs_list, output.boundary_ind_list, lb_n
            ):
                ratio_loss = ratio_loss + compute_ratio_loss(b_probs, b_inds, N)

            for s, b_ind in enumerate(output.boundary_ind_list):
                key = f"compression_L{s+1}/L{s}"
                stage_ratios[key] = compute_compression_ratio(b_ind)

            bpic = compute_bpic(L0, output.boundary_ind_list)

        val_metrics.update(
            nll_sum_nats=nll_sum.item(),
            n_tokens=B * L0,
            lb_loss=lb_loss.item(),
            ratio_loss=ratio_loss.item(),
            bpic=bpic,
            stage_ratios=stage_ratios,
        )

    model.train()
    for m in frozen_modules:
        m.eval()

    avgs = val_metrics.averages(bytes_per_token=bytes_per_token)
    if not avgs:
        return None, None, None, None, None, {}

    return (
        avgs["lm_loss"],
        avgs["lb_loss"],
        avgs["ratio_loss"],
        avgs["bpb"],
        avgs["bpic"],
        avgs["stage_ratios"],
    )
