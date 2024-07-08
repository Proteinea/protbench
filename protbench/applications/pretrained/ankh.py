from typing import List
from typing import Tuple

from peft import LoraConfig
from peft import get_peft_model
from transformers import AutoTokenizer
from transformers import T5EncoderModel
from transformers import T5ForConditionalGeneration

model_url_map = {
    "ankh-base": "ElnaggarLab/ankh-base",
    "ankh-large": "ElnaggarLab/ankh-large",
    "ankh-v2-23": "proteinea-ea/ankh-v2-large-23epochs-a3ee1d6115a726fe83f96d96f76489ff2788143c",  # noqa
    "ankh-v2-32": "proteinea-ea/ankh-v2-large-32epochs-f60c3a7c8e07fe26bdba04670ab1997f4b679969",  # noqa
    "ankh-v2-33": "proteinea-ea/ankh-v2-large-33epochs-218254e2e0546838d1427f7f6c32c0cb4664da72",  # noqa
    "ankh-v2-41": "proteinea-ea/ankh-v2-large-41epochs-e4a2c3615ff005e5e7b5bbd33ec0654106b64f1a",  # noqa
    "ankh-v2-45": "proteinea-ea/ankh-v2-large-45epochs-62fe367d20d957efdf6e8afe6ae1c724f5bc6775",  # noqa
}


def get_available_checkpoints():
    return list(model_url_map.keys())


def embeddings_postprocessing_fn(model_outputs):
    return model_outputs.last_hidden_state


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

    def __call__(self, sequence):
        output = self.tokenizer(
            sequence,
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
    if initialize_with_lora:
        model = T5ForConditionalGeneration.from_pretrained(
            model_url_map[checkpoint]
        )
        peft_config = LoraConfig(
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config).encoder
    else:
        model = T5EncoderModel.from_pretrained(model_url_map[checkpoint])

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer
