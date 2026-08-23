# Copyright (c) 2023, Tri Dao.
# Compat layer added to support GPUs below sm_80 (e.g. sm_60, sm_70), where
# the official `flash_attn` package cannot be built/run.
"""
Camada de compatibilidade para o pacote `flash_attn`.

O pacote oficial `flash_attn` só roda em GPUs com compute capability >= 8.0
(Ampere ou superior). Este módulo detecta a arquitetura da GPU em tempo de
execução e:

  - Se a GPU for sm_80+ e o pacote `flash_attn` estiver instalado, usa as
    funções reais do `flash_attn` (mesmo comportamento/performance de sempre).
  - Caso contrário (sm_60, sm_70, ou `flash_attn` não instalado), usa uma
    implementação equivalente escrita em PyTorch puro, baseada em
    `torch.nn.functional.scaled_dot_product_attention`, que funciona em
    qualquer GPU CUDA (usa os kernels "efficient"/"math" do PyTorch em vez
    dos kernels Triton/CUDA do flash_attn).

A interface (nomes de função, shapes de entrada/saída) é mantida idêntica à
do pacote `flash_attn`, então o restante do código (ex.: `CausalMHA`) não
precisa saber qual backend está sendo usado.

IMPORTANTE: o caminho de fallback é funcionalmente equivalente, mas não é
otimizado para performance/memória da mesma forma que os kernels fundidos do
flash_attn (ele materializa a matriz de atenção via SDPA do PyTorch). Isso é
esperado: GPUs sm_60/sm_70 não têm os tensor cores/recursos que o flash_attn
explora, então o objetivo aqui é corretude e compatibilidade, não paridade de
velocidade.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

__all__ = [
    "HAS_FLASH_ATTN",
    "flash_attn_qkvpacked_func",
    "flash_attn_kvpacked_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_with_kvcache",
]


# ---------------------------------------------------------------------------
# Detecção de backend
# ---------------------------------------------------------------------------

def _gpu_supports_flash_attn() -> bool:
    """flash_attn (oficial) requer compute capability >= 8.0 (sm_80+)."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _minor = torch.cuda.get_device_capability()
    except Exception:
        return False
    return major >= 8


_USE_REAL_FLASH_ATTN = False
if _gpu_supports_flash_attn():
    try:
        from flash_attn import (
            flash_attn_kvpacked_func as _real_kvpacked_func,
            flash_attn_qkvpacked_func as _real_qkvpacked_func,
            flash_attn_varlen_kvpacked_func as _real_varlen_kvpacked_func,
            flash_attn_varlen_qkvpacked_func as _real_varlen_qkvpacked_func,
            flash_attn_with_kvcache as _real_with_kvcache,
        )

        _USE_REAL_FLASH_ATTN = True
    except ImportError:
        _USE_REAL_FLASH_ATTN = False

HAS_FLASH_ATTN = _USE_REAL_FLASH_ATTN


# ---------------------------------------------------------------------------
# Utilidades comuns ao fallback
# ---------------------------------------------------------------------------

def _repeat_kv_heads(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """x: (B, S, Hkv, D) -> (B, S, Hkv * n_rep, D). Usado para GQA/MQA quando
    o número de heads de K/V é menor que o de Q."""
    if n_rep == 1:
        return x
    B, S, Hkv, D = x.shape
    return x[:, :, :, None, :].expand(B, S, Hkv, n_rep, D).reshape(B, S, Hkv * n_rep, D)


def _build_attn_mask(
    seqlen_q: int,
    seqlen_k: int,
    window_size: Tuple[int, int],
    causal: bool,
    device,
    dtype,
    query_offset: Union[int, torch.Tensor] = 0,
) -> torch.Tensor:
    """
    Constroi uma mascara aditiva (0 = permitido, valor bem negativo =
    bloqueado) equivalente a causal + sliding window do flash_attn, onde
    window_size = (left, right) e valores negativos significam "sem limite"
    naquele lado (convencao do flash_attn).

    query_offset: posicao absoluta do primeiro token de query dentro da
    sequencia de keys. Pode ser um int (mesmo offset para todo o batch) ou
    um tensor (B,) com um offset por amostra (usado no KV-cache, onde cada
    amostra pode estar em uma posicao de geracao diferente).

    Retorna shape (seqlen_q, seqlen_k) se query_offset for int, ou
    (B, seqlen_q, seqlen_k) se for tensor.
    """
    left, right = window_size
    k_idx = torch.arange(seqlen_k, device=device)
    q_pos = torch.arange(seqlen_q, device=device)

    if isinstance(query_offset, torch.Tensor):
        q_abs = query_offset.view(-1, 1) + q_pos.view(1, -1)  # (B, Sq)
        q_abs = q_abs.unsqueeze(-1)  # (B, Sq, 1)
        k_idx_b = k_idx.view(1, 1, -1)  # (1, 1, Sk)
    else:
        q_abs = (q_pos + query_offset).view(-1, 1)  # (Sq, 1)
        k_idx_b = k_idx.view(1, -1)  # (1, Sk)

    q_abs_b, k_idx_bb = torch.broadcast_tensors(q_abs, k_idx_b)
    allowed = torch.ones_like(q_abs_b, dtype=torch.bool)
    if causal:
        allowed = allowed & (k_idx_bb <= q_abs_b)
    if left is not None and left >= 0:
        allowed = allowed & (k_idx_bb >= (q_abs_b - left))
    if right is not None and right >= 0:
        allowed = allowed & (k_idx_bb <= (q_abs_b + right))

    neg = torch.finfo(dtype).min
    mask = torch.zeros_like(allowed, dtype=dtype)
    mask = mask.masked_fill(~allowed, neg)
    return mask


def _sdpa_with_padding(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_valid_mask: Optional[torch.Tensor],
    softmax_scale: Optional[float],
    causal: bool,
    window_size: Tuple[int, int],
    query_offset: Union[int, torch.Tensor] = 0,
) -> torch.Tensor:
    """
    q: (B, Sq, Hq, D); k, v: (B, Sk, Hk, D)
    key_valid_mask: (B, Sk) bool, True = posicao valida (usado para
        mascarar padding em sequencias empacotadas/varlen). None = tudo
        valido.
    Retorna: (B, Sq, Hq, D)
    """
    B, Sq, Hq, D = q.shape
    Sk = k.shape[1]
    Hk = k.shape[2]
    if Hk != Hq:
        assert Hq % Hk == 0, "num_heads de Q deve ser multiplo do de K/V"
        k = _repeat_kv_heads(k, Hq // Hk)
        v = _repeat_kv_heads(v, Hq // Hk)

    q_ = q.transpose(1, 2)  # (B, H, Sq, D)
    k_ = k.transpose(1, 2)
    v_ = v.transpose(1, 2)
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)

    use_window = (window_size[0] is not None and window_size[0] >= 0) or (
        window_size[1] is not None and window_size[1] >= 0
    )
    needs_explicit_mask = use_window or key_valid_mask is not None or isinstance(
        query_offset, torch.Tensor
    )

    if not needs_explicit_mask:
        out = F.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=None, scale=scale, is_causal=causal
        )
        return out.transpose(1, 2)

    base_mask = _build_attn_mask(
        Sq, Sk, window_size, causal, q.device, q.dtype, query_offset=query_offset
    )
    if base_mask.dim() == 2:
        base_mask = base_mask.unsqueeze(0).expand(B, Sq, Sk)
    attn_mask = base_mask.unsqueeze(1)  # (B, 1, Sq, Sk)

    if key_valid_mask is not None:
        pad = (~key_valid_mask).view(B, 1, 1, Sk)
        attn_mask = attn_mask.masked_fill(pad, torch.finfo(q.dtype).min)

    out = F.scaled_dot_product_attention(
        q_, k_, v_, attn_mask=attn_mask, scale=scale, is_causal=False
    )
    return out.transpose(1, 2)


def _unpack_varlen(x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int):
    """x: (total, H, D) -> (B, max_seqlen, H, D) com padding + mascara de validade (B, max_seqlen)."""
    B = cu_seqlens.shape[0] - 1
    H, D = x.shape[-2], x.shape[-1]
    out = x.new_zeros(B, max_seqlen, H, D)
    valid = torch.zeros(B, max_seqlen, dtype=torch.bool, device=x.device)
    cu = cu_seqlens.tolist()
    for i in range(B):
        s, e = cu[i], cu[i + 1]
        length = e - s
        if length > 0:
            out[i, :length] = x[s:e]
            valid[i, :length] = True
    return out, valid


def _pack_varlen(x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """x: (B, S, H, D) -> (total, H, D) usando cu_seqlens."""
    B = cu_seqlens.shape[0] - 1
    cu = cu_seqlens.tolist()
    pieces = []
    for i in range(B):
        s, e = cu[i], cu[i + 1]
        length = e - s
        if length > 0:
            pieces.append(x[i, :length])
    return torch.cat(pieces, dim=0) if pieces else x.new_zeros(0, x.shape[-2], x.shape[-1])


# ---------------------------------------------------------------------------
# Rotary embedding (apenas para o caminho de KV-cache fundido)
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor, interleaved: bool) -> torch.Tensor:
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rotary_raw(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool
) -> torch.Tensor:
    """
    x: (..., rot_dim)
    cos, sin: (..., rot_dim / 2), broadcastable com x exceto na ultima dim.
    """
    if not interleaved:
        cos_ = torch.cat([cos, cos], dim=-1)
        sin_ = torch.cat([sin, sin], dim=-1)
    else:
        cos_ = torch.repeat_interleave(cos, 2, dim=-1)
        sin_ = torch.repeat_interleave(sin, 2, dim=-1)
    return x * cos_ + _rotate_half(x, interleaved) * sin_


# ---------------------------------------------------------------------------
# Implementacoes de fallback (assinatura espelha o flash_attn real)
# ---------------------------------------------------------------------------

def _fallback_qkvpacked_func(qkv, softmax_scale=None, causal=False, window_size=(-1, -1)):
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    return _sdpa_with_padding(q, k, v, None, softmax_scale, causal, window_size)


def _fallback_kvpacked_func(q, kv, softmax_scale=None, causal=False, window_size=(-1, -1)):
    k, v = kv[:, :, 0], kv[:, :, 1]
    return _sdpa_with_padding(q, k, v, None, softmax_scale, causal, window_size)


def _fallback_varlen_qkvpacked_func(
    qkv, cu_seqlens, max_seqlen, softmax_scale=None, causal=False, window_size=(-1, -1)
):
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # cada uma (total, H, D)
    q_p, valid = _unpack_varlen(q, cu_seqlens, max_seqlen)
    k_p, _ = _unpack_varlen(k, cu_seqlens, max_seqlen)
    v_p, _ = _unpack_varlen(v, cu_seqlens, max_seqlen)
    out = _sdpa_with_padding(q_p, k_p, v_p, valid, softmax_scale, causal, window_size)
    return _pack_varlen(out, cu_seqlens)


def _fallback_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
):
    k, v = kv[:, 0], kv[:, 1]
    q_p, _valid_q = _unpack_varlen(q, cu_seqlens_q, max_seqlen_q)
    k_p, valid_k = _unpack_varlen(k, cu_seqlens_k, max_seqlen_k)
    v_p, _ = _unpack_varlen(v, cu_seqlens_k, max_seqlen_k)
    out = _sdpa_with_padding(q_p, k_p, v_p, valid_k, softmax_scale, causal, window_size)
    return _pack_varlen(out, cu_seqlens_q)


def _fallback_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    softmax_scale=None,
    causal=False,
    rotary_interleaved=False,
    window_size=(-1, -1),
):
    """
    Equivalente em PyTorch puro do flash_attn_with_kvcache, para o passo de
    decodificacao (poucos tokens novos por vez). Escreve k/v novos em
    k_cache/v_cache IN PLACE (mesmo comportamento do flash_attn real, ja que
    quem chama passa views do tensor de cache).

    q: (B, Sq, Hq, D)
    k_cache, v_cache: (B, max_seqlen, Hk, D)  -- mutados in place
    k, v: (B, Sq, Hk, D) novos k/v a inserir (opcional)
    cache_seqlens: int ou tensor (B,) com o numero de tokens ja validos no
        cache de cada amostra (posicao onde os novos tokens sao escritos).
    """
    B, Sq, Hq, D = q.shape
    device = q.device

    if cache_seqlens is None:
        cache_seqlens_t = torch.zeros(B, dtype=torch.long, device=device)
    elif isinstance(cache_seqlens, int):
        cache_seqlens_t = torch.full((B,), cache_seqlens, dtype=torch.long, device=device)
    else:
        cache_seqlens_t = cache_seqlens.to(device=device, dtype=torch.long)

    # 1) Aplica rotary embedding em q e no k novo, nas posicoes absolutas.
    if rotary_cos is not None and rotary_sin is not None:
        rot_dim = rotary_cos.shape[-1] * 2
        pos = cache_seqlens_t.unsqueeze(1) + torch.arange(Sq, device=device).unsqueeze(0)  # (B, Sq)
        cos_q = rotary_cos[pos].unsqueeze(2)  # (B, Sq, 1, rot_dim/2)
        sin_q = rotary_sin[pos].unsqueeze(2)

        q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
        q = torch.cat(
            [_apply_rotary_raw(q_rot, cos_q, sin_q, rotary_interleaved), q_pass], dim=-1
        )
        if k is not None:
            k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
            k = torch.cat(
                [_apply_rotary_raw(k_rot, cos_q, sin_q, rotary_interleaved), k_pass], dim=-1
            )

    # 2) Escreve k, v novos no cache (in place).
    if k is not None and v is not None:
        for i in range(B):
            start = int(cache_seqlens_t[i].item())
            end = start + Sq
            k_cache[i, start:end] = k[i]
            v_cache[i, start:end] = v[i]

    # 3) Atencao de q contra cache[:, :end] de cada amostra.
    max_end = int((cache_seqlens_t + Sq).max().item())
    k_used = k_cache[:, :max_end]
    v_used = v_cache[:, :max_end]
    key_valid = torch.arange(max_end, device=device).unsqueeze(0) < (
        cache_seqlens_t + Sq
    ).unsqueeze(1)

    out = _sdpa_with_padding(
        q,
        k_used,
        v_used,
        key_valid,
        softmax_scale,
        causal=True,
        window_size=window_size,
        query_offset=cache_seqlens_t,
    )
    return out


# ---------------------------------------------------------------------------
# API publica: seleciona automaticamente flash_attn real ou fallback
# ---------------------------------------------------------------------------

def flash_attn_qkvpacked_func(qkv, softmax_scale=None, causal=False, window_size=(-1, -1)):
    if HAS_FLASH_ATTN:
        return _real_qkvpacked_func(
            qkv, softmax_scale=softmax_scale, causal=causal, window_size=window_size
        )
    return _fallback_qkvpacked_func(qkv, softmax_scale, causal, window_size)


def flash_attn_kvpacked_func(q, kv, softmax_scale=None, causal=False, window_size=(-1, -1)):
    if HAS_FLASH_ATTN:
        return _real_kvpacked_func(
            q, kv, softmax_scale=softmax_scale, causal=causal, window_size=window_size
        )
    return _fallback_kvpacked_func(q, kv, softmax_scale, causal, window_size)


def flash_attn_varlen_qkvpacked_func(
    qkv, cu_seqlens, max_seqlen, softmax_scale=None, causal=False, window_size=(-1, -1)
):
    if HAS_FLASH_ATTN:
        return _real_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )
    return _fallback_varlen_qkvpacked_func(
        qkv, cu_seqlens, max_seqlen, softmax_scale, causal, window_size
    )


def flash_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
):
    if HAS_FLASH_ATTN:
        return _real_varlen_kvpacked_func(
            q,
            kv,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )
    return _fallback_varlen_kvpacked_func(
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
    )


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    softmax_scale=None,
    causal=False,
    rotary_interleaved=False,
    window_size=(-1, -1),
):
    if HAS_FLASH_ATTN:
        return _real_with_kvcache(
            q,
            k_cache,
            v_cache,
            k,
            v,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=causal,
            rotary_interleaved=rotary_interleaved,
            window_size=window_size,
        )
    return _fallback_with_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_seqlens=cache_seqlens,
        softmax_scale=softmax_scale,
        causal=causal,
        rotary_interleaved=rotary_interleaved,
        window_size=window_size,
    )
