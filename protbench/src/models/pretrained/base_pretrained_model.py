import abc
from typing import List

import torch


class BasePretrainedModel(abc.ABC):
    """
    Base class for pretrained models.
    """

    @abc.abstractmethod
    def embed_sequences(
        self, sequences: List[str], device: torch.device
    ) -> List[torch.Tensor]:
        pass
