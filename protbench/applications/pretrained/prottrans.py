from typing import List
from typing import Tuple

from peft import LoraConfig
from peft import get_peft_model
from transformers import AutoTokenizer
from transformers import T5EncoderModel
import re

model_url_map = {
    "prott5": "Rostlab/prot_t5_xl_uniref50",
}


def get_available_checkpoints():
    return list(model_url_map.keys())


def embeddings_postprocessing_fn(model_outputs):
    return model_outputs.last_hidden_state


def embedding_dim(model):
    return model.config.d_model


class DefaultTokenizationFunction:
    def __init__(self, tokenizer, tokenizer_options={}):
        self.tokenizer = tokenizer
        self.tokenizer_options = (
            dict(**tokenizer_options) if tokenizer_options is not None else {}
        )
        add_special_tokens = self.tokenizer_options.get(
            "add_special_tokens",
            None,
        )
        if add_special_tokens:
            self.tokenizer_options.pop("add_special_tokens", None)
        self.tokenizer_options.pop("return_tensors", None)

    def preprocess_sequences(self, sequences):
        return [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]

    def __call__(self, sequences):
        sequences = self.preprocess_sequences(sequences)
        output = self.tokenizer(
            sequences,
            add_special_tokens=True,
            return_tensors="pt",
            **self.tokenizer_options,
        )["input_ids"]
        return output


def initialize_model_from_checkpoint(
    checkpoint: str,
    initialize_with_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_bias: str = "none",
    target_modules: List = ["q", "v"],
    gradient_checkpointing: bool = False,
) -> Tuple[T5EncoderModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_url_map[checkpoint])
    model = T5EncoderModel.from_pretrained(model_url_map[checkpoint])

    if initialize_with_lora:
        peft_config = LoraConfig(
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer
