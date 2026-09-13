import re
import copy
from dataclasses import dataclass, field

import optree

from typing import Optional

import torch
import torch.nn as nn

from .flash_attn_ops_compat import RMSNorm

from hnet.modules.block import create_block
from hnet.modules.utils import get_seq_idx, get_stage_cfg

from hnet.models.config_hnet import HNetConfig


@dataclass
class IsotropicInferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[torch.Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()

        optree.tree_map(
            lambda x: x.zero_() if isinstance(x, torch.Tensor) else x,
            self.key_value_memory_dict,
        )


class Isotropic(nn.Module):
    def __init__(
        self,
        config: HNetConfig,
        pos_idx: int,
        stage_idx: int,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.stage_idx = stage_idx
        self.d_model = config.d_model[self.stage_idx]

        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]
        arch_layout = arch_layout[pos_idx]

        # ── NOVO: backbone externo pré-treinado (ex.: Llama) ──────────────────
        external_cfg = config.external_backbones.get(arch_layout)
        if external_cfg is not None:
            self._init_external(external_cfg, device=device, dtype=dtype)
            return
        self.is_external = False
        self.is_frozen_external = False

        self.ssm_cfg = get_stage_cfg(config.ssm_cfg, stage_idx)
        self.attn_cfg = get_stage_cfg(config.attn_cfg, stage_idx)
        layout_parse = re.findall(r"([mMtT])(\d+)", arch_layout)

        layers = []
        layer_idx = 0
        self.arch_full = []

        # self.height counts the number of things that get added to the residual stream
        self.height = 0
        for arch, n_layer in layout_parse:
            assert arch in ("m", "M", "t", "T")
            assert n_layer.isdigit()
            layers += [
                create_block(
                    arch,
                    self.d_model,
                    d_intermediate=config.d_intermediate[self.stage_idx],
                    ssm_cfg=self.ssm_cfg,
                    attn_cfg=self.attn_cfg,
                    layer_idx=(layer_idx + i),
                    **factory_kwargs,
                )
                for i in range(int(n_layer))
            ]
            if arch.islower():
                self.height += int(n_layer)
            else:
                self.height += 2 * int(n_layer)
            self.arch_full.extend([arch for _ in range(int(n_layer))])
            layer_idx += int(n_layer)

        self.layers = nn.ModuleList(layers)

        self.rmsnorm = RMSNorm(self.d_model, eps=1e-5, **factory_kwargs)

    def _init_external(self, external_cfg, device=None, dtype=None):
        from transformers import AutoModel

        self.is_external = True
        self.is_frozen_external = external_cfg.frozen
        self.height = 0  # não participa da contagem de residuals nativa do H-Net

        self.external_model = AutoModel.from_pretrained(
            external_cfg.hf_model, torch_dtype=next(iter([dtype])) or None
        )
        self.external_model.to(device=device, dtype=dtype)

        hf_hidden = getattr(self.external_model.config, "hidden_size", None)
        if hf_hidden != self.d_model:
            raise ValueError(
                f"d_model do estágio ({self.d_model}) != hidden_size de "
                f"'{external_cfg.hf_model}' ({hf_hidden}). Ajuste d_model no "
                f"model-config pra bater com o backbone."
            )

        if external_cfg.frozen:
            for p in self.external_model.parameters():
                p.requires_grad_(False)
            self.external_model.eval()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """
        Allocate the inference cache for the Isotropic module.

        Arguments:
            batch_size: int. The number of sequences in the batch.
            max_seqlen: int. The maximum sequence length in the batch, not used for this module.
            dtype: torch.dtype. The dtype of the inference cache.

        The inference cache contains a list of inference caches, one for each block.
        """
        if self.is_external:
            return None
        key_value_memory_dict = {}
        for i, layer in enumerate(self.layers):
            key_value_memory_dict[i] = layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype
            )
        return IsotropicInferenceParams(
            key_value_memory_dict=key_value_memory_dict,
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
        )

    def forward(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        mask=None,
        inference_params=None,
        **mixer_kwargs,
    ):
        if self.is_external:
            return self._forward_external(hidden_states, cu_seqlens, mask)
        assert (mask is not None) or (
            cu_seqlens is not None and max_seqlen is not None
        ), "Either mask or cu_seqlens and max_seqlen must be provided"

        attn_mixer_kwargs = copy.deepcopy(mixer_kwargs)
        ssm_mixer_kwargs = copy.deepcopy(mixer_kwargs)
        if mask is not None:
            packed = False
            assert (
                hidden_states.dim() == 3
            ), "Hidden states must be (B, L, D) in unpacked mode"
        else:
            attn_mixer_kwargs.update(
                {"cu_seqlens": cu_seqlens.int(), "max_seqlen": max_seqlen}
            )
            ssm_mixer_kwargs.update(
                {"seq_idx": get_seq_idx(cu_seqlens, device=hidden_states.device)}
            )
            packed = True

        residual = None
        for layer, arch in zip(self.layers, self.arch_full):
            if arch in ("m", "M"):
                layer_mixer_kwargs = ssm_mixer_kwargs
                if hidden_states.dim() == 2:
                    hidden_states = hidden_states.unsqueeze(0)
                    residual = None if residual is None else residual.unsqueeze(0)
            elif arch in ("t", "T"):
                layer_mixer_kwargs = attn_mixer_kwargs
                if hidden_states.dim() == 3 and packed:
                    hidden_states = hidden_states.squeeze(0)
                    residual = None if residual is None else residual.squeeze(0)
            else:
                # Currently supporting only Mamba2 and MHA
                raise NotImplementedError

            hidden_states, residual = layer(
                hidden_states,
                residual,
                inference_params=inference_params,
                mixer_kwargs=layer_mixer_kwargs,
            )

        # Setting prenorm=False ignores the residual
        hidden_states = self.rmsnorm(
            hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
        )

        if hidden_states.dim() == 3 and packed:
            hidden_states = hidden_states.squeeze(0)

        if inference_params is not None:
            # here we also explicitly assume the mask is all True
            assert mask.shape[0] == 1, "seqlen_offset handling assumes batch size 1"
            inference_params.seqlen_offset += hidden_states.shape[1]

        return hidden_states

    def _forward_external(self, hidden_states, cu_seqlens, mask):
        if self.is_external:
            raise NotImplementedError(
                "Geração passo-a-passo com backbone externo ainda não suportada "
                "(precisaria do KV-cache nativo do HF, não do IsotropicInferenceParams)."
            )
        if mask is not None:
            # modo unpacked: já é (B, L, D)
            assert hidden_states.dim() == 3
            attn_mask = mask.to(dtype=torch.long)
            out = self.external_model(inputs_embeds=hidden_states, attention_mask=attn_mask)
            return out.last_hidden_state

        # modo packed: (total_tokens, D) + cu_seqlens -> despacota, roda, repacota
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        chunks = torch.split(hidden_states, seq_lens, dim=0)
        padded = torch.nn.utils.rnn.pad_sequence(chunks, batch_first=True)  # (B, Lmax, D)

        B, Lmax = padded.shape[0], padded.shape[1]
        attn_mask = torch.zeros(B, Lmax, dtype=torch.long, device=hidden_states.device)
        for i, L in enumerate(seq_lens):
            attn_mask[i, :L] = 1

        out = self.external_model(inputs_embeds=padded, attention_mask=attn_mask).last_hidden_state
        return torch.cat([out[i, :L] for i, L in enumerate(seq_lens)], dim=0)

    def step(self, hidden_states, inference_params):
        """
        Assumes hidden_states is (B, 1, D). Steps each of the layers in order, and then steps the main model.
        """
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer.step(
                hidden_states, inference_params, residual=residual
            )

        hidden_states = self.rmsnorm(
            hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
        )
        inference_params.seqlen_offset += 1

        return hidden_states
