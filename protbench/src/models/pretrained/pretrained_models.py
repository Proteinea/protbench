from typing import List, Dict

import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from torch.utils.data import DataLoader
from protbench.src.models.pretrained import BasePretrainedModel
from protbench.src.models.model_registry import PretrainedModelRegistry
from protbench.src.models.pretrained.util_datasets import SequencesDataset


@PretrainedModelRegistry.register("huggingface")
class HuggingFaceModels(BasePretrainedModel):
    def __init__(
        self,
        model_url: str,
        batch_size: int = 1,
        num_workers: int = 1,
        hf_kwargs: Dict = {},
    ):
        """A template class for loading pretrained models and tokenizers from huggingface.

        Args:
            model_url (str): pretrained model/tokenizer name or path on huggingface. See pretrained_model_name_or_path
                in https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained
                The model and tokenizer are loaded using transformers.AutoModel/AutoTokenizer.
            batch_size (int): batch size used to batch sequences to extract embeddings. Defaults to 1.
            num_workers (int): number of workers used in dataloader. Defaults to 1.
            hf_kwargs (Dict): additional keyword arguments passed to AutoModel/AutoTokenizer.from_pretrained.
                Defaults to empty dict.
        """
        self.model = self.load_model(model_url, hf_kwargs)
        self.tokenizer = self.load_tokenizer(model_url, hf_kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def load_model(self, model_url: str, hf_kwargs: Dict = {}) -> torch.nn.Module:
        model = AutoModel.from_pretrained(model_url, **hf_kwargs)
        return model

    def load_tokenizer(
        self, tokenizer_url: str, hf_kwargs: Dict = {}
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_url, **hf_kwargs)
        return tokenizer

    def collate_data(self, batch: List[str]) -> BatchEncoding:
        return self.tokenizer(
            batch,
            padding="longest",
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
        )

    def get_dataloader(self, sequences: List[str]) -> DataLoader:
        sequences_dataset = SequencesDataset(sequences)
        sequences_dataloader = DataLoader(
            sequences_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_data,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )
        return sequences_dataloader

    @torch.inference_mode()
    def embed_sequences(
        self, sequences: List[str], device: torch.device
    ) -> List[torch.Tensor]:
        sequences_dataloader = self.get_dataloader(sequences)
        self.model.eval()
        self.model.to(device)
        embeddings = []
        with tqdm(
            sequences_dataloader, desc="Embedding sequences", unit="batch", ascii=" ="
        ) as tqdm_dataloader:
            for batch in tqdm_dataloader:
                batch = batch.to(device)
                embds_batch = self.model(**batch)[0].to("cpu")
                # below snippet is to extract the actual embeddings without padding
                for attention_mask, embds in zip(
                    batch["attention_mask"].to("cpu"), embds_batch
                ):
                    embeddings.append(embds[attention_mask == 1])

        self.model.to("cpu")
        return embeddings

    def get_number_of_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


@PretrainedModelRegistry.register("t5_based")
class T5BasedModels(HuggingFaceModels):
    def __init__(
        self,
        model_url: str,
        batch_size: int = 1,
        num_workers: int = 1,
        hf_kwargs: Dict = {},
    ):
        """Load T5 based model by loading the model's encoder.

        Args:
            model_url (str): pretrained model/tokenizer name or path on huggingface. See pretrained_model_name_or_path
                in https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained
                The model and tokenizer are loaded using transformers.AutoModel/AutoTokenizer.
            batch_size (int): batch size used to batch sequences to extract embeddings. Defaults to 1.
            num_workers (int): number of workers used in dataloader. Defaults to 1.
            hf_kwargs (Dict): additional keyword arguments passed to AutoModel/AutoTokenizer.from_pretrained.
                Defaults to empty dict.
        """
        super(T5BasedModels, self).__init__(
            model_url, batch_size, num_workers, hf_kwargs
        )

    def load_model(self, model_url: str, hf_kwargs: Dict = {}) -> torch.nn.Module:
        return super(T5BasedModels, self).load_model(model_url, hf_kwargs).get_encoder()
