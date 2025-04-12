"""
2025.3.17
2025.3.19
4.51.2
0.15.2
__UNSLOTH_VERSIONING__
"""

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import importlib.util
if importlib.util.find_spec("unsloth_studio") is None:
    UNSLOTH_STUDIO_ENABLED = False
else:
    UNSLOTH_STUDIO_ENABLED = os.environ.get("UNSLOTH_STUDIO_DISABLED", "0") == "0"
pass
from typing import List, Dict, Tuple, Optional, Any, Callable
import math


import os
import torch
from unsloth_zoo.loss_utils import fused_linear_cross_entropy

if UNSLOTH_STUDIO_ENABLED:
    from unsloth_zoo.loss_utils import fast_linear_cross_entropy

scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
@torch.compiler.disable(recursive = False)
def disable_compile_scaled_dot_product_attention(*args, **kwargs):
    return scaled_dot_product_attention(*args, **kwargs)
pass


torch_compile_options = {'epilogue_fusion': True, 'max_autotune': False, 'shape_padding': True, 'trace.enabled': False, 'triton.cudagraphs': False}

from torch.nn import CrossEntropyLoss

@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def normal_cross_entropy_loss(self, hidden_states, labels):
    logits = self.lm_head(hidden_states)
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, self.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss, logits
pass

# We need an empty logits flag to warn people logits will not be returned anymore unless asked ie
# os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
LOGITS_ERROR_STRING = \
    "Unsloth: Logits are empty from 2024.11 onwards. To get raw logits again, please "\
    'set the environment variable `UNSLOTH_RETURN_LOGITS` to `"1" BEFORE starting to train ie before `trainer.train()`. For example:\n'\
    "```\nimport os\n"\
    "os.environ['UNSLOTH_RETURN_LOGITS'] = '1'\n"\
    "trainer.train()\n```\n"\
    "No need to restart your console - just add `os.environ['UNSLOTH_RETURN_LOGITS'] = '1'` before trainer.train() and re-run the cell!"

def raise_logits_error(*args, **kwargs): raise NotImplementedError(LOGITS_ERROR_STRING)
def return_none(*args, **kwargs): return None
class EmptyLogits:
    def __init__(self): return
    def raise_getattr_error(self, attr): return return_none if attr == "to" else raise_logits_error
    __getitem__ = raise_logits_error
    __getattr__ = raise_getattr_error
    def __repr__(self): return LOGITS_ERROR_STRING
    def __str__ (self): return LOGITS_ERROR_STRING
pass
EMPTY_LOGITS = EmptyLogits()
functions = dir(torch.Tensor)
for j, function in enumerate(functions):
    if function.startswith("__") and function.endswith("__"):
        exec(f"def raise_{j}(*args, **kwargs): print('{function}')", globals(), locals())
        try: exec(f"EMPTY_LOGITS.{function} = raise_{j}", globals(), locals())
        except: continue
pass


from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (math, Callable, Optional, Tuple, Union, torch, nn, F, ACT2FN, Cache, GenerationMixin, FlashAttentionKwargs, BaseModelOutputWithPast, CausalLMOutputWithPast, ROPE_INIT_FUNCTIONS, dynamic_rope_update, ALL_ATTENTION_FUNCTIONS, PreTrainedModel, Unpack, add_start_docstrings, add_start_docstrings_to_model_forward, can_return_tuple, replace_return_docstrings, deprecate_kwarg, DeepseekV3Config, logger, __name__, _CONFIG_FOR_DOC, DeepseekV3PreTrainedModel, DEEPSEEK_V3_INPUTS_DOCSTRING, DeepseekV3Model, KwargsForCausalLM, DeepseekV3ForCausalLM)

@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def DeepseekV3RMSNorm_forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)

class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return DeepseekV3RMSNorm_forward(self, hidden_states)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
@torch.no_grad()
@dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
def DeepseekV3RotaryEmbedding_forward(self, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, config: DeepseekV3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq


    def forward(self, x, position_ids):
        return DeepseekV3RotaryEmbedding_forward(self, x, position_ids)


@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def DeepseekV3MLP_forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

class DeepseekV3MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return DeepseekV3MLP_forward(self, x)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def DeepseekV3TopkRouter_forward(self, hidden_states):
    hidden_states = hidden_states.view(-1, self.config.hidden_size)
    router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
    scores = router_logits.sigmoid()
    topk_indices = self.get_topk_indices(scores)
    topk_weights = scores.gather(1, topk_indices)
    if self.norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights /= denominator
    topk_weights = topk_weights * self.routed_scaling_factor
    return topk_indices, topk_weights

class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros((self.n_routed_experts)))

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        return DeepseekV3TopkRouter_forward(self, hidden_states)


@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def DeepseekV3MoE_forward(self, hidden_states):
    residuals = hidden_states
    orig_shape = hidden_states.shape
    topk_indices, topk_weights = self.gate(hidden_states)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
    hidden_states = hidden_states + self.shared_experts(residuals)
    return hidden_states

class DeepseekV3MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [
                DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = DeepseekV3TopkRouter(config)
        self.shared_experts = DeepseekV3MLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        r"""
        CALL FOR CONTRIBUTION! I don't have time to optimise this right now, but expert weights need to be fused
        to not have to do a loop here (deepseek has 256 experts soooo yeah).
        """
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)

        # in original deepseek, the output of the experts are gathered once we leave this module
        # thus the moe module is itelsf an IsolatedParallel module
        # and all expert are "local" meaning we shard but we don't gather
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        return DeepseekV3MoE_forward(self, hidden_states)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def apply_rotary_pos_emb(q, k, cos, sin,  unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def apply_rotary_pos_emb_interleave(q, k, cos, sin,  unsqueeze_dim=1):
    r"""
    TODO let's just use the original freqcis computation to not have the view
    transpose + reshape! This is not optimized!
    Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


@torch.compiler.disable(recursive = False)
def DeepseekV3Attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
    key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

    q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states))).view(query_shape).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

    k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
    k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

    k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

    cos, sin = position_embeddings
    if self.config.rope_interleave:  # support using interleaved weights for efficiency
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    else:
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
        value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
        attn_output = attn_output[:, :, :, : self.v_head_dim]

    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

class DeepseekV3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim

        self.is_causal = True
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV3RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.scaling = self.qk_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        return DeepseekV3Attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)


@torch.compiler.disable(recursive = False)
@can_return_tuple
@deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
@add_start_docstrings_to_model_forward(DEEPSEEK_V3_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def DeepseekV3ForCausalLM_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs: Unpack[KwargsForCausalLM],
) -> CausalLMOutputWithPast:
    r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        logits_to_keep (`int` or `torch.Tensor`, *optional*):
            If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
            If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
            This is useful when using packed tensor format (single dimension for batch and sequence length).

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, DeepseekV3ForCausalLM

    >>> model = DeepseekV3ForCausalLM.from_pretrained("meta-deepseek_v3/DeepseekV3-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-deepseek_v3/DeepseekV3-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs: BaseModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = EMPTY_LOGITS
    loss = None
    NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
    n_items = (kwargs).get("num_items_in_batch", None) or (kwargs).get("n_items", None)
    requires_grad_ = self.lm_head.weight.requires_grad
    
    if labels is None:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
    elif (UNSLOTH_STUDIO_ENABLED and NOT_RETURN_LOGITS and labels is not None) and not requires_grad_:
        loss = fast_linear_cross_entropy(
            hidden_states        = hidden_states[:, slice_indices, :],
            lm_head              = self.lm_head,
            labels               = labels,
            num_items_in_batch   = n_items,
            logit_softcapping    = None if () == () else (),
            logit_scale_multiply = None if () == () else (),
            logit_scale_divide   = None if () == () else (),
        )
    elif (() == () and () == ()) and NOT_RETURN_LOGITS and self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None and not requires_grad_:
        loss = fused_linear_cross_entropy(
            hidden_states      = hidden_states[:, slice_indices, :],
            lm_weight          = self.lm_head.weight,
            labels             = labels.to(self.lm_head.weight.device),
            num_items_in_batch = n_items,
            logit_softcapping  = None if () == () else (),
        )
    elif self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        def _compiled_loss_function(
            output_logits : torch.Tensor,
            output_labels : torch.Tensor,
            logit_scale_multiply : float = 0,
            logit_scale_divide : float = 0,
            logit_softcapping : float = 0,
            vocab_size : int = 0,
            n_items : int = 0,
        ):
            device = output_logits.device
            if logit_scale_multiply != 0:
                output_logits = output_logits * logit_scale_multiply
            if logit_scale_divide != 0:
                output_logits = output_logits / logit_scale_divide
            if logit_softcapping != 0:
                output_logits = output_logits / logit_softcapping
                output_logits = torch.tanh(output_logits)
                output_logits = output_logits * logit_softcapping
    
            shift_logits = output_logits
            shift_labels = torch.empty_like(output_labels, device = device)
            shift_labels[..., :-1] = output_labels[..., 1:]
            shift_labels[..., -1] = -100
            # shift_logits = output_logits[..., :-1, :].float().contiguous()
            # shift_labels = output_labels[..., 1:].contiguous()
    
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)
    
            n_chunks = int(math.ceil((vocab_size / 262144) * 8))
            if requires_grad_: n_chunks += 2
            __shift_logits = torch.chunk(shift_logits, n_chunks, dim = 0)
            __shift_labels = torch.chunk(shift_labels, n_chunks, dim = 0)
            loss = 0.0
            for (_shift_logits, _shift_labels) in zip(__shift_logits, __shift_labels):
                loss += torch.nn.functional.cross_entropy(
                    input  = _shift_logits.float().contiguous(),
                    target = _shift_labels.contiguous(),
                    reduction = 'sum',
                )
            pass
            if n_items != 0:
                loss = loss / n_items
            else:
                loss = loss / (shift_labels != -100).sum()
            return loss
        pass
        _compiled_loss_function = torch.compile(
            _compiled_loss_function,
            fullgraph = False,
            dynamic = True,
            options = torch_compile_options,
        )
        torch._dynamo.mark_dynamic(logits, 1)
        torch._dynamo.mark_dynamic(labels, 1)
        loss = _compiled_loss_function(
            output_logits        = logits,
            output_labels        = labels,
            logit_scale_multiply = () if () != () else 0,
            logit_scale_divide   = () if () != () else 0,
            logit_softcapping    = () if () != () else 0,
            vocab_size = (self.config.vocab_size),
            n_items              = n_items if n_items is not None else 0,
        )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if () != ():
            logits = logits / ()
            logits = torch.tanh(logits)
            logits = logits * ()
        loss = self.loss_function(logits=logits, labels=labels.to(self.lm_head.weight.device), vocab_size=self.config.vocab_size, **kwargs)


    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

class DeepseekV3ForCausalLM(DeepseekV3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekV3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        return DeepseekV3ForCausalLM_forward(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, cache_position, logits_to_keep, **kwargs)
