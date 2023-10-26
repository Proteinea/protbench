from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5EncoderModel,
)
from peft import get_peft_model, LoraConfig, TaskType
from typing import Tuple


model_url_map = {
    "ankh-base": "ElnaggarLab/ankh-base",
    "ankh-large": "ElnaggarLab/ankh-large",
    "ankh-v2-23": "proteinea-ea/ankh-v2-large-23epochs-a3ee1d6115a726fe83f96d96f76489ff2788143c",
    "ankh-v2-32": "proteinea-ea/ankh-v2-large-32epochs-f60c3a7c8e07fe26bdba04670ab1997f4b679969",
    "ankh-v2-33": "proteinea-ea/ankh-v2-large-33epochs-218254e2e0546838d1427f7f6c32c0cb4664da72",
    "ankh-v2-41": "proteinea-ea/ankh-v2-large-41epochs-e4a2c3615ff005e5e7b5bbd33ec0654106b64f1a",
    "ankh-v2-45": "proteinea-ea/ankh-v2-large-45epochs-62fe367d20d957efdf6e8afe6ae1c724f5bc6775",
}


def get_available_checkpoints():
    return list(model_url_map.keys())


def initialize_model_from_checkpoint(
    model_name: str,
    initialize_with_lora: bool = False,
    task_type: TaskType = None,
) -> Tuple[T5EncoderModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_url_map[model_name])
    if initialize_with_lora:
        model = T5ForConditionalGeneration.from_pretrained(
            model_url_map[model_name]
        )
        peft_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, peft_config).encoder
    else:
        model = T5EncoderModel.from_pretrained(model_url_map[model_name])
    return model, tokenizer
