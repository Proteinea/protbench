from typing import List
from peft import LoraConfig
from peft import get_peft_model
from transformers import AutoModel
from transformers import AutoTokenizer

model_url_map = {
    "esm2_650M": "facebook/esm2_t33_650M_UR50D",
    "esm2_3B": "facebook/esm2_t36_3B_UR50D",
    "esm2_15B": "facebook/esm2_t48_15B_UR50D",
}


class DefaultTokenizationFunction:
    def __init__(self, tokenizer, tokenizer_options={}):
        self.tokenizer = tokenizer
        self.tokenizer_options = (
            tokenizer_options if tokenizer_options is not None else {}
        )
        add_special_tokens = self.tokenizer_options.get(
            "add_special_tokens",
            None,
        )
        if add_special_tokens:
            self.tokenizer_options.pop("add_special_tokens")
        self.tokenizer_options.pop("return_tensors")

    def __call__(self, sequences):
        return self.tokenizer(
            sequences,
            add_special_tokens=True,
            return_tensors="pt",
            **self.tokenizer_options,
        )["input_ids"]


def embeddings_postprocessing_fn(model_outputs):
    return model_outputs.last_hidden_state


def initialize_model_from_checkpoint(
    checkpoint,
    initialize_with_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_bias: str = "none",
    target_modules: List = ["q", "v"],
    gradient_checkpointing: bool = False,
):
    model = AutoModel.from_pretrained(model_url_map[checkpoint])
    tokenizer = AutoTokenizer.from_pretrained(model_url_map[checkpoint])

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
