from protbench import applications
from typing import List


AVAILABLE_MODELS = [
    "esm2",
    "ankh",
    "prottrans",
]


def get_model_module(model_family):
    global AVAILABLE_MODELS
    if model_family not in AVAILABLE_MODELS:
        raise ValueError("Expected model family to be one of the "
                         f"following: {AVAILABLE_MODELS}. "
                         f"Received: {model_family}.")
    module = getattr(applications.pretrained, model_family)
    return module


def initialize_model_from_checkpoint(
    model_family: str,
    checkpoint: str,
    initialize_with_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_bias: str = "none",
    target_modules: List = ["q", "v"],
    gradient_checkpointing: bool = False,
):
    module = getattr(applications.pretrained, model_family)
    pretrained_model, tokenizer = module.initialize_model_from_checkpoint(
        checkpoint,
        initialize_with_lora=initialize_with_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_bias=lora_bias,
        target_modules=target_modules,
        gradient_checkpointing=gradient_checkpointing,
    )
    return pretrained_model, tokenizer
