import math

import torch
import torch.nn as nn

from typing import Optional, AnyStr

from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import InferenceParams
from .pom import evit, pumer, merge_only, prune_only, hybrid


class PoMMamba(nn.Module):
    sparse_mask = None

    def __init__(self,
                 layer: Block,
                 layer_index: int,
                 reduce_ratio: float = 1,
                 metrics: Optional[AnyStr] = 'clip',
                 hidden_state_strategy: Optional[AnyStr] = 'prune',
                 residual_strategy: Optional[AnyStr] = 'prune',
                 preserve_length: Optional[bool] = True):
        super().__init__()
        self.layer = layer
        self.layer_index = layer_index
        self.reduce_ratio = reduce_ratio
        self.metrics = metrics
        self.hidden_state_strategy = hidden_state_strategy
        self.residual_strategy = residual_strategy
        self.preserve_length = preserve_length

        # self.method_func = dict(
        #     prune=self.prune,
        #     merge=self.merge,
        # )

    @classmethod
    def from_layer(cls, layer, *args, **kwargs):
        return cls(layer=layer, *args, **kwargs)

    def forward(self,
                hidden_states: torch.Tensor,
                residual: torch.Tensor,
                inference_params: Optional[InferenceParams] = None,
                **kwargs,
                ):

        hidden_states, residual = self.layer(
            hidden_states, residual, inference_params=inference_params
        )
        with torch.no_grad():
            attn_heads = hidden_states.detach()
            if self.metrics == 'clip':
                attn_heads = attn_heads.clamp(min=0)
                avg_heads = (attn_heads.sum(dim=-1)).detach()
            else:
                avg_heads = (attn_heads.sum(dim=-1)).detach()
            avg_heads[avg_heads == 0] = float('-inf')

        B, N, D = hidden_states.shape
        num_keep_node = math.ceil(N * self.reduce_ratio)

        if self.hidden_state_strategy == 'evit':
            hidden_states = evit(hidden_states, avg_heads, num_keep_node,
                                 preserve_length=self.preserve_length, obj=self)
        elif self.hidden_state_strategy == 'pumer':
            hidden_states = pumer(hidden_states, attn_heads, num_keep_node,
                                  preserve_length=self.preserve_length, obj=self)
        elif self.hidden_state_strategy == 'prune_only':
            hidden_states = prune_only(hidden_states, attn_heads, num_keep_node,
                                         preserve_length=self.preserve_length, obj=self)
        elif self.hidden_state_strategy == 'merge_only':
            hidden_states = merge_only(hidden_states, attn_heads, num_keep_node,
                                         preserve_length=self.preserve_length, obj=self)
        elif self.hidden_state_strategy == 'hybrid':
            hidden_states = hybrid(hidden_states, attn_heads, num_keep_node,
                                        preserve_length=self.preserve_length, obj=self)
        else:
            NotImplementedError(self.hidden_state_strategy)

        if self.residual_strategy == 'evit':
            residual = evit(residual, avg_heads, num_keep_node,
                            preserve_length=self.preserve_length, obj=self)
        elif self.residual_strategy == 'pumer':
            residual = pumer(residual, attn_heads, num_keep_node,
                             preserve_length=self.preserve_length, obj=self)
        elif self.residual_strategy == 'prune_only':
            residual = prune_only(residual, attn_heads, num_keep_node,
                                    preserve_length=self.preserve_length, obj=self)
        elif self.residual_strategy == 'merge_only':
            residual = merge_only(residual, attn_heads, num_keep_node,
                                    preserve_length=self.preserve_length, obj=self)
        elif self.residual_strategy == 'hybrid':
            residual = hybrid(residual, attn_heads, num_keep_node,
                                   preserve_length=self.preserve_length, obj=self)
        else:
            NotImplementedError(self.residual_strategy)

        self.sparse_mask = None
        return (
            hidden_states, residual
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.layer.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
