import torch
import torch.nn.functional as F

from hnet.models.mixer_seq import find_frozen_external_modules
from hnet.utils.metrics import HNetMetrics, compute_bpic, compute_compression_ratio, compute_ratio_loss


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
    Validação com a mesma lógica de métricas do train_pt.py.

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

        # NLL exata (reduction="sum") -> BPB exato sobre todo o split
        nll_sum = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
            reduction="sum",
        )

        ratio_loss   = torch.zeros((), device=device)
        stage_ratios = {}
        bpic         = 0.0

        if output.bpred_output and lb_n:
            boundary_ind_list = []
            for s, (bpred, N) in enumerate(zip(output.bpred_output, lb_n)):
                boundary_probs = bpred.boundary_prob[..., 1]
                boundary_ind   = bpred.boundary_mask.float()
                boundary_ind_list.append(boundary_ind)

                ratio_loss = ratio_loss + compute_ratio_loss(boundary_probs, boundary_ind, N)
                stage_ratios[f"compression_L{s+1}/L{s}"] = compute_compression_ratio(boundary_ind)

            bpic = compute_bpic(L0, boundary_ind_list)

        # igual ao train_pt.py: lb_loss passa a ser o ratio_loss
        lb_loss = ratio_loss

        val_metrics.update(
            nll_sum_nats=nll_sum.item(),
            n_tokens=B * L0,
            lb_loss=lb_loss.item(),
            ratio_loss=ratio_loss.item(),
            bpic=bpic,
            stage_ratios=stage_ratios,
        )

    # o eval_pt.py chama model.eval() por conta própria; o train_pt.py usa o
    # modelo em modo treino depois da validação
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