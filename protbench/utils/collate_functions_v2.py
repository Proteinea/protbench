import abc
import warnings
from typing import Any, Dict, List

import torch


def warn_experimental(cls_name):
    warnings.warn(
        f"{cls_name} is still experimental and not ready "
        "for production usage.",
        UserWarning,
    )


class CollatorFunction(abc.ABC):
    def __init__(self, tokenizer=None) -> None:
        self.tokenizer = tokenizer
        warn_experimental(self.__class__.__name__)

    def merge_dictionary_keys(self, inputs: List[Dict]) -> Dict:
        outputs = {}
        for current_input in inputs:
            for k, v in current_input.items():
                if k in outputs:
                    outputs[k].append(v)
                else:
                    outputs[k] = [v]

    @abc.abstractmethod
    def call(self, batch: Dict) -> Dict:
        pass

    def __call__(self, batch) -> Any:
        batch = self.merge_dictionary_keys(batch)
        return self.call(batch)


class CollateEmbeddingsAndLabels(CollatorFunction):
    def __init__(
        self, padding_value=0, pad_labels=False, label_padding_value=-100
    ) -> None:
        super().__init__(tokenizer=None)
        self.padding_value = padding_value
        self.pad_labels = pad_labels
        self.label_padding_value = label_padding_value

    def call(self, batch: Dict) -> Dict:
        embeddings = torch.nn.utils.rnn.pad_sequence(
            batch["embeddings"],
            batch_first=True,
            padding_value=self.padding_value,
        )
        if not isinstance(batch["labels"], torch.Tensor):
            labels = torch.tensor(batch["labels"])

        if self.pad_labels:
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=self.label_padding_value
            )
        return {"embeddings": embeddings, "labels": labels}
