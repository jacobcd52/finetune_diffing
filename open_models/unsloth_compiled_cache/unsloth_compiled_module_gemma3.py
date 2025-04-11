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
from transformers.models.gemma3.modeling_gemma3 import (copy, List, Optional, Tuple, Union, torch, nn, ACT2FN, Cache, HybridCache, StaticCache, GenerationMixin, BaseModelOutputWithPast, ModelOutput, CausalLMOutputWithPast, ROPE_INIT_FUNCTIONS, dynamic_rope_update, PreTrainedModel, add_start_docstrings, add_start_docstrings_to_model_forward, can_return_tuple, is_torchdynamo_compiling, replace_return_docstrings, deprecate_kwarg, AutoModel, AutoModelForCausalLM, Gemma3Config, Gemma3TextConfig, logger, __name__, _CONFIG_FOR_DOC, Gemma3CausalLMOutputWithPast, GEMMA3_START_DOCSTRING, Gemma3PreTrainedModel, GEMMA3_INPUTS_DOCSTRING, Gemma3TextModel, Gemma3ForCausalLM, Gemma3ForConditionalGeneration)

@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def Gemma3MLP_forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        return Gemma3MLP_forward(self, x)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def Gemma3RMSNorm_forward(self, x):
    output = self._norm(x.float())
    # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
    # See https://github.com/huggingface/transformers/pull/29402
    output = output * (1.0 + self.weight.float())
    return output.type_as(x)

class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return Gemma3RMSNorm_forward(self, x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
@torch.no_grad()
@dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
def Gemma3RotaryEmbedding_forward(self, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Gemma3RotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma3TextConfig, device=None):
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
        return Gemma3RotaryEmbedding_forward(self, x, position_ids)


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
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


@torch.compiler.disable(recursive = False)
@can_return_tuple
@deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
@add_start_docstrings_to_model_forward(GEMMA3_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def Gemma3ForCausalLM_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[HybridCache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **loss_kwargs,
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
    >>> from transformers import AutoTokenizer, Gemma3ForCausalLM

    >>> model = Gemma3ForCausalLM.from_pretrained("google/gemma-2-9b")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

    >>> prompt = "What is your favorite condiment?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "What is your favorite condiment?"
    ```"""

    if self.training and self.config._attn_implementation != "eager":
        logger.warning_once(
            "It is strongly recommended to train Gemma3 models with the `eager` attention implementation "
            f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
        )
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
        **loss_kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = EMPTY_LOGITS
    loss = None
    NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
    n_items = (loss_kwargs).get("num_items_in_batch", None) or (loss_kwargs).get("n_items", None)
    requires_grad_ = self.lm_head.weight.requires_grad
    
    if labels is None:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
    elif (UNSLOTH_STUDIO_ENABLED and NOT_RETURN_LOGITS and labels is not None) and not requires_grad_:
        loss = fast_linear_cross_entropy(
            hidden_states        = hidden_states[:, slice_indices, :],
            lm_head              = self.lm_head,
            labels               = labels,
            num_items_in_batch   = n_items,
            logit_softcapping    = None if (self.config.final_logit_softcapping) == () else (self.config.final_logit_softcapping),
            logit_scale_multiply = None if () == () else (),
            logit_scale_divide   = None if () == () else (),
        )
    elif (() == () and () == ()) and NOT_RETURN_LOGITS and self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None and not requires_grad_:
        loss = fused_linear_cross_entropy(
            hidden_states      = hidden_states[:, slice_indices, :],
            lm_weight          = self.lm_head.weight,
            labels             = labels.to(self.lm_head.weight.device),
            num_items_in_batch = n_items,
            logit_softcapping  = None if (self.config.final_logit_softcapping) == () else (self.config.final_logit_softcapping),
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
            logit_softcapping    = (self.config.final_logit_softcapping) if (self.config.final_logit_softcapping) != () else 0,
            vocab_size           = (self.vocab_size),
            n_items              = n_items if n_items is not None else 0,
        )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if (self.config.final_logit_softcapping) != ():
            logits = logits / (self.config.final_logit_softcapping)
            logits = torch.tanh(logits)
            logits = logits * (self.config.final_logit_softcapping)
        loss = self.loss_function(logits, labels.to(self.lm_head.weight.device), self.vocab_size, **loss_kwargs)


    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

class Gemma3ForCausalLM(Gemma3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config_class = Gemma3TextConfig
    base_model_prefix = "language_model"

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.model = Gemma3TextModel(config)
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
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> CausalLMOutputWithPast:
        return Gemma3ForCausalLM_forward(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, cache_position, logits_to_keep, **loss_kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten: has a special cache type, `HybridCache`

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # This is needed to correctly slice the mask without data-dependent slicing later on if using dynamo tracing
        # (retrieving the same value from `cache_position` later on would crash dynamo)
        model_inputs["last_cache_position"] = attention_mask.shape[-1] if attention_mask is not None else 0
        if logits_to_keep is None:
            _ = model_inputs.pop("logits_to_keep", None)

        if (
            isinstance(past_key_values, HybridCache)
            and attention_mask.ndim == 2
            and not self.config._attn_implementation == "flash_attention_2"
        ):
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
            )
            model_inputs["attention_mask"] = attention_mask

        return model_inputs


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def Gemma3MultiModalProjector_forward(self, vision_outputs: torch.Tensor):
    batch_size, _, seq_length = vision_outputs.shape

    reshaped_vision_outputs = vision_outputs.transpose(1, 2)
    reshaped_vision_outputs = reshaped_vision_outputs.reshape(
        batch_size, seq_length, self.patches_per_image, self.patches_per_image
    )
    reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

    pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
    pooled_vision_outputs = pooled_vision_outputs.flatten(2)
    pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

    normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

    projected_vision_outputs = torch.matmul(normed_vision_outputs, self.mm_input_projection_weight)
    return projected_vision_outputs.type_as(vision_outputs)

class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(config.vision_config.hidden_size, config.text_config.hidden_size)
        )

        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor):
        return Gemma3MultiModalProjector_forward(self, vision_outputs)


@torch.compiler.disable(recursive = False)
@can_return_tuple
@deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
@add_start_docstrings_to_model_forward(GEMMA3_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=Gemma3CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def Gemma3ForConditionalGeneration_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **lm_kwargs,
) -> Union[Tuple, Gemma3CausalLMOutputWithPast]:
    r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

        logits_to_keep (`int` or `torch.Tensor`, *optional*):
            If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
            If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
            This is useful when using packed tensor format (single dimension for batch and sequence length).

    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    >>> model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-it")
    >>> processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

    >>> messages = [
    ...     {
    ...         "role": "system",
    ...         "content": [
    ...             {"type": "text", "text": "You are a helpful assistant."}
    ...         ]
    ...     },
    ...     {
    ...         "role": "user", "content": [
    ...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
    ...             {"type": "text", "text": "Where is the cat standing?"},
    ...         ]
    ...     },
    ... ]

    >>> inputs = processor.apply_chat_template(
    ...     messages,
    ...     tokenizer=True,
    ...     return_dict=True,
    ...     return_tensors="pt",
    ...     add_generation_prompt=True
    ... )
    >>> # Generate
    >>> generate_ids = model.generate(**inputs)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "user\nYou are a helpful assistant.\n\n\n\n\n\nWhere is the cat standing?\nmodel\nBased on the image, the cat is standing in a snowy area, likely outdoors. It appears to"
    ```
    """

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    is_training = token_type_ids is not None and labels is not None

    # Replace image id woth PAD if the image token if OOV, to avoid index-errors
    if input_ids is not None and self.config.image_token_index >= self.vocab_size:
        special_image_mask = input_ids == self.config.image_token_index
        llm_input_ids = input_ids.clone()
        llm_input_ids[special_image_mask] = 0
    else:
        llm_input_ids = input_ids

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(llm_input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # Merge text and images
    if pixel_values is not None:
        image_features = self.get_image_features(pixel_values)

        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_index, dtype=torch.long, device=inputs_embeds.device)
            )
        else:
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

        if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
            image_tokens_in_text = (special_image_mask).sum(dim=1).sum(dim=0)[0]
            raise ValueError(
                f"Number of images does not match number of special image tokens in the input text. "
                f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                "tokens from image embeddings."
            )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    # mask out pad-token-ids in labels for BC
    if labels is not None and self.pad_token_id in labels:
        logger.warning_once(
            "`labels` contains `pad_token_id` which will be masked with `config.ignore_index`. "
            "You have to mask out `pad_token_id` when preparing `labels`, this behavior will be removed in v.4.46.",
        )
        labels = torch.where(input_ids == self.pad_token_id, -100, labels)

    causal_mask = self._update_causal_mask(
        attention_mask, token_type_ids, past_key_values, cache_position, inputs_embeds, is_training
    )
    outputs: CausalLMOutputWithPast = self.language_model(
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        **lm_kwargs,
    )

    logits = outputs.logits
    loss = None
    NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
    
    all_locals = locals()
    n_items = None
    for __kwargs in all_locals.values():
        if type(__kwargs) is dict:
            n_items = __kwargs.get("num_items_in_batch", None) or __kwargs.get("n_items", None)
            break
    
    if labels is not None:
        def _compiled_loss_function(
            output_logits : torch.Tensor,
            output_labels : torch.Tensor,
            mask : torch.Tensor = None,
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
            if mask is not None:
                mask = mask.to(device = device)
                shift_labels[..., :-1][mask[..., 1:] == 0] = -100
            pass
            shift_labels[..., -1] = -100
    
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)
    
            __shift_logits = torch.chunk(shift_logits, 4, dim = 0)
            __shift_labels = torch.chunk(shift_labels, 4, dim = 0)
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
        if attention_mask is not None:
            torch._dynamo.mark_dynamic(attention_mask, 1)
        loss = _compiled_loss_function(
            output_logits        = logits,
            output_labels        = labels,
            mask                 = attention_mask,
            logit_scale_multiply = () if () != () else 0,
            logit_scale_divide   = () if () != () else 0,
            logit_softcapping    = () if () != () else 0,
            vocab_size           = (self.config.text_config.vocab_size),
            n_items              = n_items if n_items is not None else 0,
        )


    return Gemma3CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
    )

@add_start_docstrings(
    """The GEMMA3 model which consists of a vision backbone and a language model.""",
    GEMMA3_START_DOCSTRING,
)
class Gemma3ForConditionalGeneration(Gemma3PreTrainedModel, GenerationMixin):
    def __init__(self, config: Gemma3Config):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config=config.vision_config)
        self.multi_modal_projector = Gemma3MultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size

        language_model = AutoModelForCausalLM.from_config(config=config.text_config)

        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids,
        past_key_values,
        cache_position,
        input_tensor,
        is_training: bool = False,
    ):
        if self.config.text_config._attn_implementation == "flash_attention_2":
            return attention_mask

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted
            # form and requires no inversion or slicing.
            return attention_mask

        using_static_cache = isinstance(past_key_values, StaticCache)
        min_dtype = torch.finfo(self.dtype).min
        inputs_lead_dim, sequence_length = input_tensor.shape[:2]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        elif isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            return attention_mask

        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=self.dtype, device=cache_position.device
        )

        # Causal diagonal mask only if training, otherwise attend to the whole prefix. Training-specific attn for prefix is handled below
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)

        # Apply bidirectional mask on images if token type ids are provided
        if token_type_ids is not None and sequence_length != 1:
            token_type_mask = token_type_ids.unsqueeze(1) == token_type_ids.unsqueeze(2)
            token_type_mask[token_type_ids == 0] = False  # if text token do not change anything
            token_type_mask = token_type_mask.unsqueeze(1).to(causal_mask.device, dtype=torch.bool)
            causal_mask = causal_mask.clone()
            causal_mask[:, :, :, :sequence_length] = causal_mask[:, :, :, :sequence_length].masked_fill(
                token_type_mask, 0.0
            )

        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]

            # Then apply padding mask (will mask pad tokens)
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

        return causal_mask

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Projects the last hidden state from the vision model into language model space.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_outputs = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        image_features = self.multi_modal_projector(vision_outputs)
        return image_features


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Union[Tuple, Gemma3CausalLMOutputWithPast]:
        return Gemma3ForConditionalGeneration_forward(self, input_ids, pixel_values, attention_mask, position_ids, past_key_values, token_type_ids, cache_position, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, logits_to_keep, **lm_kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model. NOTE: use_cache=False needs pixel_values always
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
        is_training = token_type_ids is not None and labels is not None
        if cache_position[0] == 0 and isinstance(past_key_values, HybridCache):
            input_tensor = inputs_embeds if inputs_embeds is not None else input_ids
            causal_mask = self._update_causal_mask(
                attention_mask, token_type_ids, past_key_values, cache_position, input_tensor, is_training
            )
            model_inputs["attention_mask"] = causal_mask

        return model_inputs

    def tie_weights(self):
        return self.language_model.tie_weights()
