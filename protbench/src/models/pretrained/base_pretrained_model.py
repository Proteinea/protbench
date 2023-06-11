import abc
from typing import List

import torch


class BasePretrainedModel(abc.ABC):
    """
    Base class for pretrained models.
    """

    def freeze_model(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    @abc.abstractmethod
    def embed_sequences(
        self, sequences: List[str], device: torch.device
    ) -> List[torch.Tensor]:
        pass
