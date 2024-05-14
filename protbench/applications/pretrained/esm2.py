from typing import List
from typing import Literal

import esm
from peft import LoraConfig
from peft import TaskType
from peft import get_peft_model

model_url_map = {
    "esm2_650M": esm.pretrained.esm2_t33_650M_UR50D,
    "esm2_3B": esm.pretrained.esm2_t36_3B_UR50D,
    "esm2_t48_15B_UR50D": esm.pretrained.esm2_t48_15B_UR50D,
}


class DefaultTokenizationFunction:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sequence):
        _, _, encoded_sequence = self.tokenizer(
            [("sequence", sequence)]
        )
        return encoded_sequence


def embeddings_postprocessing_fn(
    output,
    repr_layer: int | Literal["last"] = "last"
):
    if repr_layer == "last":
        key = sorted(list(output["representations"].keys()))[-1]
    else:
        key = repr_layer
    return output["representations"][key]


def initialize_model_from_checkpoint(
        checkpoint,
        initialize_with_lora: bool = False,
        lora_task_type: TaskType = None,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_bias: str = "none",
        target_modules: List = ["q", "k"],
        gradient_checkpointing: bool = False,):
    model, alphabet = model_url_map[checkpoint]()
    batch_converter = alphabet.get_batch_converter()

    if initialize_with_lora:
        peft_config = LoraConfig(
            task_type=lora_task_type,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)

    if gradient_checkpointing:
        raise ValueError("ESM does not support gradient checkpointing.")

    return model, batch_converter
