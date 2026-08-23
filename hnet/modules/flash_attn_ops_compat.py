# Compat layer for flash_attn ops that depend on Triton, plus pass-through
# re-exports for the ops that don't.
"""
Este módulo cobre os quatro imports adicionais do flash_attn:

    from flash_attn.ops.triton.rotary import apply_rotary
    from flash_attn.ops.activations import swiglu
    from flash_attn.ops.triton.layer_norm import RMSNorm
    from flash_attn.utils.generation import GenerationMixin

Cada um tem uma relação diferente com a arquitetura da GPU:

  - `apply_rotary` e `RMSNorm` (em `flash_attn.ops.triton.*`) usam kernels
    escritos na linguagem/compilador Triton (OpenAI Triton), que exige GPU
    com compute capability >= 7.0. Isso cobre sm_70 (Volta) mas NÃO cobre
    sm_60 (Pascal). Este módulo detecta isso e cai para uma implementação em
    PyTorch puro quando necessário (equivalente à função de referência
    `apply_rotary_emb_torch` que o próprio flash_attn expõe para CPU/testes,
    e a uma normalização RMS/Layer padrão do PyTorch).

  - `swiglu` (`flash_attn.ops.activations`) já é implementado em PyTorch
    puro no pacote original (não usa Triton nem CUDA custom kernel), então
    não depende da arquitetura da GPU. Fornecido aqui apenas por
    completude/independência do pacote `flash_attn`.

  - `GenerationMixin` (`flash_attn.utils.generation`) é lógica Python pura
    (laço de geração/decodificação), sem kernels CUDA/Triton, então também
    não depende da arquitetura da GPU. Reexportado sem modificação (com
    fallback em caso de o pacote `flash_attn` não estar instalado).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

__all__ = [
    "HAS_TRITON_OPS",
    "apply_rotary",
    "swiglu",
    "RMSNorm",
    "layer_norm_fn",
    "GenerationMixin",
]


# ---------------------------------------------------------------------------
# Deteccao de suporte a Triton
# ---------------------------------------------------------------------------

def _gpu_supports_triton() -> bool:
    """OpenAI Triton (usado pelos kernels flash_attn.ops.triton.*) exige
    compute capability >= 7.0 (Volta ou superior). sm_60 (Pascal) fica de
    fora; sm_70 (Volta) e superiores sao suportados."""
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return False
    return (major, minor) >= (7, 0)


_USE_TRITON_ROTARY = False
_USE_TRITON_LAYER_NORM = False

if _gpu_supports_triton():
    try:
        from flash_attn.ops.triton.rotary import apply_rotary as _real_apply_rotary

        _USE_TRITON_ROTARY = True
    except ImportError:
        _USE_TRITON_ROTARY = False

    try:
        from flash_attn.ops.triton.layer_norm import (
            RMSNorm as _RealRMSNorm,
            layer_norm_fn as _real_layer_norm_fn,
        )

        _USE_TRITON_LAYER_NORM = True
    except ImportError:
        _USE_TRITON_LAYER_NORM = False

HAS_TRITON_OPS = _USE_TRITON_ROTARY and _USE_TRITON_LAYER_NORM


# ---------------------------------------------------------------------------
# swiglu -- ja e puro PyTorch no pacote original, sem dependencia de arch.
# Reimplementado aqui para nao depender do import de `flash_attn` funcionar.
# ---------------------------------------------------------------------------

def swiglu(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU: recebe a projecao combinada (..., 2 * d) e retorna (..., d).
    Convencao: primeira metade = gate, segunda metade = valor.
    out = silu(gate) * valor
    """
    x1, x2 = x.chunk(2, dim=-1)
    return F.silu(x1) * x2


# ---------------------------------------------------------------------------
# apply_rotary -- fallback em PyTorch puro (sm_60 / sem Triton)
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor, interleaved: bool) -> torch.Tensor:
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def _apply_rotary_raw(x, cos, sin, interleaved):
    """x: (..., rot_dim); cos, sin: (..., rot_dim/2), broadcastable com x."""
    if not interleaved:
        cos_ = torch.cat([cos, cos], dim=-1)
        sin_ = torch.cat([sin, sin], dim=-1)
    else:
        cos_ = torch.repeat_interleave(cos, 2, dim=-1)
        sin_ = torch.repeat_interleave(sin, 2, dim=-1)
    return x * cos_ + _rotate_half(x, interleaved) * sin_


def _fallback_apply_rotary(
    x,
    cos,
    sin,
    seqlen_offsets=0,
    cu_seqlens=None,
    max_seqlen=None,
    interleaved=False,
    inplace=False,
    conjugate=False,
):
    """
    Equivalente em PyTorch puro de flash_attn.ops.triton.rotary.apply_rotary.

    x: (batch, seqlen, nheads, headdim) se cu_seqlens is None,
       ou (total_nnz, nheads, headdim) (sequencias empacotadas/varlen) se
       cu_seqlens for fornecido.
    cos, sin: (seqlen_tabela, rotary_dim / 2) -- tabela de rotary
        pre-computada, indexada pela posicao absoluta do token.
    seqlen_offsets: int ou tensor (batch,) -- deslocamento de posicao por
        amostra (usado no KV-cache durante geracao).
    cu_seqlens: (batch + 1,) ou None -- limites das sequencias empacotadas.
    interleaved: estilo GPT-J (pares intercalados) vs GPT-NeoX (metades).
    inplace: se True, escreve o resultado em x e retorna x.
    conjugate: se True, inverte o sinal de sin (usado no backward).
    """
    if conjugate:
        sin = -sin

    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1], "rotary_dim nao pode exceder head_dim"

    if cu_seqlens is None:
        B, S = x.shape[0], x.shape[1]
        if isinstance(seqlen_offsets, torch.Tensor):
            pos = seqlen_offsets.view(-1, 1) + torch.arange(S, device=x.device).view(1, -1)
        else:
            pos = (torch.arange(S, device=x.device) + seqlen_offsets).view(1, -1).expand(B, -1)
        cos_ = cos[pos].unsqueeze(2)  # (B, S, 1, ro_dim/2)
        sin_ = sin[pos].unsqueeze(2)
    else:
        num_seqs = cu_seqlens.shape[0] - 1
        cu = cu_seqlens.tolist()
        offs = (
            seqlen_offsets.tolist()
            if isinstance(seqlen_offsets, torch.Tensor)
            else [seqlen_offsets] * num_seqs
        )
        pos_pieces = []
        for i in range(num_seqs):
            length = cu[i + 1] - cu[i]
            if length > 0:
                pos_pieces.append(torch.arange(length, device=x.device) + offs[i])
        pos = (
            torch.cat(pos_pieces)
            if pos_pieces
            else torch.zeros(0, dtype=torch.long, device=x.device)
        )
        cos_ = cos[pos].unsqueeze(1)  # (total_nnz, 1, ro_dim/2)
        sin_ = sin[pos].unsqueeze(1)

    x_rot, x_pass = x[..., :ro_dim], x[..., ro_dim:]
    out_rot = _apply_rotary_raw(x_rot, cos_, sin_, interleaved)
    out = torch.cat([out_rot, x_pass], dim=-1)

    if inplace:
        x.copy_(out)
        return x
    return out


def apply_rotary(
    x,
    cos,
    sin,
    seqlen_offsets=0,
    cu_seqlens=None,
    max_seqlen=None,
    interleaved=False,
    inplace=False,
    conjugate=False,
):
    if _USE_TRITON_ROTARY:
        return _real_apply_rotary(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
            conjugate=conjugate,
        )
    return _fallback_apply_rotary(
        x, cos, sin, seqlen_offsets, cu_seqlens, max_seqlen, interleaved, inplace, conjugate
    )


# ---------------------------------------------------------------------------
# RMSNorm / layer_norm_fn -- fallback em PyTorch puro (sm_60 / sem Triton)
# ---------------------------------------------------------------------------

def _rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(dtype)


def _fallback_layer_norm_fn(
    x,
    weight,
    bias,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    eps=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    residual_in_fp32=False,
    is_rms_norm=False,
    return_dropout_mask=False,
    out=None,
    residual_out=None,
):
    """
    Equivalente em PyTorch puro de flash_attn.ops.triton.layer_norm.layer_norm_fn.

    Cobre bem o caso comum (norma unica, com/sem residual e dropout,
    pre-norm ou pos-norm). O caminho de "residual paralelo" (quando `x1` /
    `weight1` sao fornecidos, estilo GPT-J/GPT-NeoX/PaLM) tambem e
    implementado, mas de forma mais simples que o kernel fundido original;
    vale a pena validar numericamente se for usar esse caminho.
    """

    def _prep(inp):
        mask = None
        v = inp
        if dropout_p > 0.0:
            mask = torch.empty_like(v, dtype=torch.bool).bernoulli_(1 - dropout_p)
            v = v.masked_fill(~mask, 0) / (1 - dropout_p)
        if rowscale is not None:
            v = v * rowscale.unsqueeze(-1)
        return v, mask

    x_, mask_x = _prep(x)
    x1_, mask_x1 = (_prep(x1) if x1 is not None else (None, None))

    compute_dtype = torch.float32 if residual_in_fp32 else x_.dtype
    combined = x_.to(compute_dtype)
    if residual is not None:
        combined = combined + residual.to(compute_dtype)
    if x1_ is not None:
        combined = combined + x1_.to(compute_dtype)

    residual_out_val = combined if prenorm else None

    def _norm(inp, w, b):
        if is_rms_norm:
            return _rms_norm_ref(inp, w, eps)
        return F.layer_norm(
            inp.float(), (inp.shape[-1],), w.float(), b.float() if b is not None else None, eps
        ).to(w.dtype)

    out_val = _norm(combined.to(weight.dtype), weight, bias)

    result = [out_val]
    if weight1 is not None:
        out1_val = _norm(combined.to(weight1.dtype), weight1, bias1)
        result.append(out1_val)
    if prenorm:
        result.append(residual_out_val)
    if return_dropout_mask:
        result.append(mask_x)
        if x1 is not None:
            result.append(mask_x1)

    return tuple(result) if len(result) > 1 else result[0]


def layer_norm_fn(
    x,
    weight,
    bias,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    eps=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    residual_in_fp32=False,
    is_rms_norm=False,
    return_dropout_mask=False,
    out=None,
    residual_out=None,
):
    if _USE_TRITON_LAYER_NORM:
        return _real_layer_norm_fn(
            x,
            weight,
            bias,
            residual=residual,
            x1=x1,
            weight1=weight1,
            bias1=bias1,
            eps=eps,
            dropout_p=dropout_p,
            rowscale=rowscale,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=is_rms_norm,
            return_dropout_mask=return_dropout_mask,
            out=out,
            residual_out=residual_out,
        )
    return _fallback_layer_norm_fn(
        x,
        weight,
        bias,
        residual=residual,
        x1=x1,
        weight1=weight1,
        bias1=bias1,
        eps=eps,
        dropout_p=dropout_p,
        rowscale=rowscale,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
        is_rms_norm=is_rms_norm,
        return_dropout_mask=return_dropout_mask,
        out=out,
        residual_out=residual_out,
    )


class _RMSNormFallback(nn.Module):
    """Fallback em PyTorch puro para flash_attn.ops.triton.layer_norm.RMSNorm."""

    def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)

    def forward(self, x, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False):
        return layer_norm_fn(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            dropout_p=dropout_p,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=True,
        )

    def reset_parameters(self):
        nn.init.ones_(self.weight)


if _USE_TRITON_LAYER_NORM:
    RMSNorm = _RealRMSNorm
else:
    RMSNorm = _RMSNormFallback


# ---------------------------------------------------------------------------
# GenerationMixin -- logica pura Python/PyTorch, sem dependencia de arch.
# Reexportado sem modificacao.
# ---------------------------------------------------------------------------

try:
    from flash_attn.utils.generation import GenerationMixin  # noqa: E402,F401
except ImportError:
    GenerationMixin = None  # type: ignore
    # `GenerationMixin` nao tem dependencia de arquitetura de GPU -- se essa
    # importacao falhar, o problema e o pacote `flash_attn` nao estar
    # instalado, e nao a compute capability da GPU.
