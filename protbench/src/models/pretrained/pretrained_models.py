from typing import List, Optional

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
from protbench.src.models.model_registry import ModelRegistry
from protbench.src.models.pretrained.util_datasets import SequencesDataset


@ModelRegistry.register_pretrained("huggingface")
class HuggingfaceModels(BasePretrainedModel):
    def __init__(
        self,
        model_url: str,
        use_auth_token: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 1,
    ):
        """A template class for loading pretrained models and tokenizers from huggingface.

        Args:
            model_url (str): pretrained model name or path on huggingface. See pretrained_model_name_or_path
                in https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained
            use_auth_token (Optional[str], optional): hugginface authentication token. Defaults to None.
            batch_size (int): batch size used to batch sequences to extract embeddings. Defaults to 1.
        """
        self.model = self.load_model(model_url, use_auth_token).get_encoder()  # type: ignore
        self.freeze_model(self.model)
        self.tokenizer = self.load_tokenizer(model_url, use_auth_token)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def load_model(
        self, model_url: str, use_auth_token: Optional[str] = None
    ) -> torch.nn.Module:
        model = AutoModel.from_pretrained(model_url, use_auth_token=use_auth_token)
        return model

    def load_tokenizer(
        self, tokenizer_url: str, use_auth_token: Optional[str] = None
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_url, use_auth_token=use_auth_token
        )
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
