import numpy as np
import torch

from protbench.src.embedder import EmbeddingFunction


class TorchEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: torch.nn.Module, tokenizer, device):
        super().__init__(model, tokenizer)
        if self.model.training:
            self.model.eval()
        self.device = device

    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    def call(self, sequence: str) -> torch.Tensor:
        tokenized_data = self.tokenizer(sequence)
        with torch.no_grad():
            return self.model(tokenized_data.to(self.device))[0].cpu()
