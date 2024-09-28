from typing import List, AnyStr, Optional
from collections import namedtuple

import torch
import numpy as np

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from .modules.mamba_wrap import PoMMamba
from .utils.logger import get_logger

CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])


class PoMMambaLMHeadModel(MambaLMHeadModel):
    _wrap_layer = dict()
    logger = get_logger(__name__)
    preserve_length = None
    return_mask = False

    def apply_layer(self,
                    hidden_state_strategy: str = 'prune',
                    residual_strategy: str = 'prune',
                    reduce_ratio=0.925,
                    reduce_anchor: List = (20, 30, 40, 50),
                    metrics: Optional[AnyStr] = None,
                    preserve_length: bool = False,
                    return_mask: bool = False,):

        self.preserve_length = preserve_length
        self.return_mask = return_mask
        base_reduce_ratio = reduce_ratio if not preserve_length else 1
        overall_ratio = 1
        overall_sparsity = []
        effective_layer = []
        for idx, layer in enumerate(self.backbone.layers):
            overall_sparsity.append(overall_ratio)
            if idx in reduce_anchor:
                overall_ratio *= float(reduce_ratio)
                if preserve_length:
                    base_reduce_ratio *= float(reduce_ratio)

                self._wrap_layer[idx] = layer
                self.backbone.layers[idx] = PoMMamba(layer=layer,
                                                     layer_index=idx,
                                                     reduce_ratio=base_reduce_ratio,
                                                     metrics=metrics,
                                                     hidden_state_strategy=hidden_state_strategy,
                                                     residual_strategy=residual_strategy,
                                                     preserve_length=preserve_length)
        self.logger.info(self.config)
        self.logger.info(f"#####################################################")
        self.logger.info(f"### Pruning Ratio - {reduce_ratio}")
        self.logger.info(f"### Pruning Layer idx - {list(self._wrap_layer.keys())}")
        self.logger.info(f"### Hidden state Strategy - {hidden_state_strategy}")
        self.logger.info(f"### Residual Strategy - {residual_strategy}")
        self.logger.info(f"### Layer index - {reduce_anchor}")
        self.logger.info(f"### Metrics - {metrics}")
        self.logger.info(f"### Preserve Length - {preserve_length}")
        self.logger.info(f"### Model Sparsity - {1 - np.mean(overall_sparsity): .02%}")
        self.logger.info(f"#####################################################")

    def unwrap_layer(self):
        for idx, layer in enumerate(self.backbone.layers):
            if self._wrap_layer.get(idx) is not None:
                self.backbone.layers[idx] = self._wrap_layer.get(idx)

        self.logger.info(f"Unwrap {self.__class__.__name__}")

    def forward(self, *args, **kwargs):
        output: CausalLMOutput = super().forward(*args, **kwargs)
        if not self.preserve_length and self.return_mask:
            token_prune_idx_all = []
            output = super().forward(*args, **kwargs)
            for idx, layer in enumerate(self.backbone.layers):
                if isinstance(layer, PoMMamba) and layer.sparse_mask is not None:
                    token_remove_idx = torch.where((layer.sparse_mask != 0).all(dim=-1))
                    token_prune_idx_all.append(token_remove_idx[-1].tolist())

            return output.logits, token_prune_idx_all
        return output
