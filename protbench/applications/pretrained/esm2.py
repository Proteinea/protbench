from typing import List

from peft import LoraConfig
from peft import get_peft_model
from transformers import AutoModel
from transformers import AutoTokenizer

from protbench.applications.pretrained.pretrained import PretrainedModelWrapper

model_url_map = {
    "esm2_650M": "facebook/esm2_t33_650M_UR50D",
    "esm2_3B": "facebook/esm2_t36_3B_UR50D",
    "esm2_15B": "facebook/esm2_t48_15B_UR50D",
}


def get_available_checkpoints():
    return list(model_url_map.keys())


def embedding_dim(model):
    return model.config.hidden_size


class DefaultTokenizationFunction:
    def __init__(
        self, tokenizer, return_input_ids_only=False, tokenizer_options={}
    ):
        self.tokenizer = tokenizer
        self.tokenizer_options = (
            dict(**tokenizer_options) if tokenizer_options is not None else {}
        )
        self.add_special_tokens = self.tokenizer_options.pop(
            "add_special_tokens", True
        )
        self.return_tensors = self.tokenizer_options.pop(
            "return_tensors", "pt"
        )
        self.is_split_into_words = self.tokenizer_options.pop(
            "is_split_into_words", False
        )
        self.return_input_ids_only = return_input_ids_only

    def __call__(self, sequences):
        if not isinstance(sequences, list):
            sequences = [sequences]
        output = self.tokenizer(
            sequences,
            add_special_tokens=self.add_special_tokens,
            return_tensors=self.return_tensors,
            is_split_into_words=self.is_split_into_words,
            **self.tokenizer_options,
        )
        return output["input_ids"] if self.return_input_ids_only else output


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


class ESM2(PretrainedModelWrapper):
    def __init__(self, checkpoint):
        super().__init__(checkpoint=checkpoint)

    def initialze_model_from_checkpoint(
        self,
        gradient_checkpointing: bool = False,
    ):
        self.model, self.tokenizer = initialize_model_from_checkpoint(
            checkpoint=self.checkpoint,
            initialize_with_lora=False,
            gradient_checkpointing=gradient_checkpointing,
        )

        self._initialized = True

    def initialze_model_from_checkpoint_with_lora(
        self,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_bias: str,
        target_modules: List,
        gradient_checkpointing: bool = False,
    ):
        self.model, self.tokenizer = initialize_model_from_checkpoint(
            checkpoint=self.checkpoint,
            initialize_with_lora=True,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_bias=lora_bias,
            target_modules=target_modules,
            gradient_checkpointing=gradient_checkpointing,
        )
        self._initialized = True

    @property
    def embedding_dim(self):
        if not self._initialized:
            raise ValueError("Model not initialized.")
        return embedding_dim(self.model)

    @staticmethod
    def embeddings_postprocessing_fn(model_outputs):
        return embeddings_postprocessing_fn(model_outputs=model_outputs)

    @staticmethod
    def get_available_checkpoints():
        return get_available_checkpoints()

    def load_default_tokenization_function(
        self, return_input_ids_only=False, tokenizer_options={}
    ):
        return DefaultTokenizationFunction(
            self.tokenizer, return_input_ids_only, tokenizer_options
        )


def load_model_wrapper(checkpoint):
    return ESM2(checkpoint)
