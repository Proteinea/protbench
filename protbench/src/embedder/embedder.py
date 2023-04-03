import abc
from tqdm.auto import tqdm
import torch
from pathlib import Path
import numpy as np
from protbench.src import utils
from protbench.src.tasks import Task


@utils.mark_experimental(use_instead=None)
class EmbeddingFunction(abc.ABC):
    def __init__(self, model: torch.nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @abc.abstractmethod
    def call(self, data):
        raise NotImplementedError

    def __call__(self, data):
        return self(data)


@utils.mark_experimental(use_instead=None)
class TorchEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: torch.nn.Module, tokenizer):
        super().__init__(model, tokenizer)
        if self.model.training:
            self.model.eval()

    def call(self, data):
        tokenized_data = self.tokenizer(data)
        with torch.no_grad():
            return self.model(tokenized_data)


@utils.mark_experimental(use_instead=None)
class Embedder:
    def __init__(self, embedding_function, save_path: str = None):
        self.embedding_function = embedding_function
        self.save_path = Path(save_path) or None
        if self.save_path is None:
            self.embeddings = []
        else:
            self.save_path.mkdir(exist_ok=True)

    def embed_data(self, task: Task):
        sequences = task.load_sequences()
        for idx, example in enumerate(tqdm(data)):
            embedding = self.embedding_function(example)
            if self.save_path is not None:
                np.save(self.save_path / Path(str(idx)), embedding)
            else:
                self.embeddings.append(embedding)
