import abc
from typing import List

from protbench import applications


class PretrainedModelWrapper(abc.ABC):
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self._initialized = False
        self.model = None
        self.tokenizer = None

    @abc.abstractmethod
    def initialze_model_from_checkpoint(
        self,
        gradient_checkpointing: bool = False,
    ):
        pass

    def initialze_model_from_checkpoint_with_lora(
        self,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_bias: str,
        target_modules: List,
        gradient_checkpointing: bool = False,
    ):
        pass

    @abc.abstractmethod
    def embedding_dim(self):
        pass

    @abc.abstractmethod
    def embeddings_postprocessing_fn(self, model_outputs):
        pass

    def load_default_tokenization_function(
        self, return_input_ids_only=False, tokenizer_options={}
    ):
        pass


AVAILABLE_MODELS = [
    "esm2",
    "ankh",
    "prottrans",
]


def get_model_module(model_family):
    global AVAILABLE_MODELS
    if model_family not in AVAILABLE_MODELS:
        raise ValueError(
            "Expected model family to be one of the "
            f"following: {AVAILABLE_MODELS}. "
            f"Received: {model_family}."
        )
    module = getattr(applications.pretrained, model_family)
    return module


def initialize_model_from_checkpoint(
    model_family: str,
    checkpoint: str,
) -> PretrainedModelWrapper:
    module = getattr(applications.pretrained, model_family)
    if not hasattr(module, "load_model_wrapper"):
        raise ValueError(
            f"Expected module {model_family} to have `load_model_wrapper` "
            "function, this function should just return an instance of a "
            "subclass that inherits from `PretrainedModelWrapper`."
        )
    instance = module.load_model_wrapper(
        checkpoint,
    )
    return instance
