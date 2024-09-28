import torch

import transformers
from transformers import AutoTokenizer

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate

from pom.mamba_model import PoMMambaLMHeadModel
from pom.utils.logger import get_logger

from eval.huggingface import HFLM


@register_model("mamba")
class MambaEvalWrapper(HFLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
    logger = get_logger()
    seq_reduction = None


    def __init__(self,
                 pretrained="state-spaces/mamba-2.8b",
                 max_length=2048,
                 batch_size=None,
                 device="cuda",
                 hidden_state_strategy='hybrid',
                 residual_strategy='merge_only',
                 reduce_ratio=0.905,
                 reduce_anchor="12-17-22-27-32-37-42",
                 metrics='clip',
                 preserve_length=False,
                 prune=True,
                 dtype=torch.float16):
        LM.__init__(self)
        self.logger.info(f"<<<{pretrained}>>>")
        reduce_anchor = list(map(lambda x: eval(x), reduce_anchor.split('-')))
        if reduce_ratio < 1:
            if prune:
                self._model = PoMMambaLMHeadModel.from_pretrained(pretrained, device=device, dtype=dtype)

                metrics = metrics if metrics != "None" else None

                self._model.apply_layer(reduce_ratio=reduce_ratio,
                                        metrics=metrics,
                                        hidden_state_strategy=hidden_state_strategy,
                                        residual_strategy=residual_strategy,
                                        reduce_anchor=reduce_anchor,
                                        preserve_length=preserve_length)
            else:
                self.seq_reduction = reduce_ratio ** len(reduce_anchor)
                self.logger.info(f"Evaluate seq_reduction {self.seq_reduction}")
                self._model = MambaLMHeadModel.from_pretrained(pretrained, device=device, dtype=dtype)
        else:
            self._model = MambaLMHeadModel.from_pretrained(pretrained, device=device, dtype=dtype)

        with torch.no_grad():
            fake_tokens = torch.randint(1000, (1, 1000), device=device)
            hidden_states = self._model(fake_tokens).logits
            self.logger.info(f"Profile Model")
            self.logger.info(f"Input: {fake_tokens.shape}")
            self.logger.info(f"Output: {hidden_states.shape}")
            if hidden_states.size(1) == fake_tokens.size(1):
                self.logger.info(
                    f"Output Sparsity: {torch.sum(hidden_states.sum(-1) == 0) / hidden_states.sum(-1).numel():.02%}")
            else:
                self.logger.info(f"Output Sparsity: {hidden_states.size(1) / fake_tokens.size(1):.02%}")

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError()


if __name__ == "__main__":
    cli_evaluate()
