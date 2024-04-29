from functools import partial
import os
from typing import Callable, Dict, List, Optional
from protbench.embedder import TorchEmbedder
from protbench.embedder import TorchEmbeddingFunction
import torch
from dataclasses import dataclass


@dataclass
class SaveDirectories:
    parent_dir: str = "/root/.cache"
    train_dir: str = "train_embeddings"
    validation_dir: str = "validation_embeddings"
    test_dir: str = "test_embeddings"

    @property
    def train(self):
        return os.path.join(self.parent_dir, self.train_dir)

    @property
    def validation(self):
        return os.path.join(self.parent_dir, self.validation_dir)

    @property
    def test(self):
        return os.path.join(self.parent_dir, self.test_dir)


def delete_directory_contents(directory_path: str):
    for root, _, files in os.walk(directory_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))


def compute_embeddings(
    model: torch.nn.Module,
    tokenizer: Callable,
    train_seqs: List[str],
    val_seqs: Optional[List[str]] = None,
    test_seqs: Optional[List[str]] = None,
    post_processing_fn: Callable = None,
    device: Optional[torch.device] = None,
    pad_token_id: int = 0,
    tokenizer_options: Dict = {},
):
    embedding_fn = TorchEmbeddingFunction(
        model,
        partial(tokenizer, **tokenizer_options),
        device=device,
        embeddings_postprocessing_fn=post_processing_fn,
        pad_token_id=pad_token_id,
    )
    embedder = TorchEmbedder(
        embedding_fn,
        low_memory=False,
        save_path=None,
        devices=None,
        batch_size=1,
    )

    embeddings = []
    compute_data = [
        train_seqs,
    ]

    if val_seqs is not None:
        compute_data.append(val_seqs)

    if test_seqs is not None:
        compute_data.append(test_seqs)

    for data in [train_seqs, val_seqs, test_seqs]:
        if data is not None:
            embeddings.append(embedder.run(data))
        else:
            embeddings.append(None)

    return embeddings


def compute_embeddings_and_save_to_disk(
    model: torch.nn.Module,
    tokenizer: Callable,
    save_directories: SaveDirectories,
    train_seqs: List[str],
    val_seqs: Optional[List[str]] = None,
    test_seqs: Optional[List[str]] = None,
    post_processing_fn: Callable = None,
    device: Optional[torch.device] = None,
    pad_token_id: int = 0,
    tokenizer_options: Dict = {},
):
    embedding_fn = TorchEmbeddingFunction(
        model,
        partial(tokenizer, **tokenizer_options),
        device=device,
        embeddings_postprocessing_fn=post_processing_fn,
        pad_token_id=pad_token_id,
    )

    if not os.path.exists(save_directories.train):
        os.mkdir(save_directories.train)
    else:
        delete_directory_contents(save_directories.train)

    if val_seqs is not None and not os.path.exists(
        save_directories.validation
    ):
        os.mkdir(save_directories.validation)
    else:
        delete_directory_contents(save_directories.validation)

    if test_seqs is not None and not os.path.exists(save_directories.test):
        os.mkdir(save_directories.test)
    else:
        delete_directory_contents(save_directories.test)

    compute_data = [
        (train_seqs, save_directories.train),
    ]

    if val_seqs is not None:
        compute_data.append((val_seqs, save_directories.validation))

    if test_seqs is not None:
        compute_data.append((test_seqs, save_directories.test))

    for data, path in compute_data:
        embedder = TorchEmbedder(
            embedding_fn,
            low_memory=True,
            save_path=path,
            devices=None,
            batch_size=1,
        )
        embedder.run(data)


@dataclass
class EmbeddingsContainer:
    train_embeddings: List[torch.Tensor]
    val_embeddings: Optional[List[torch.Tensor]] = None
    test_embeddings: Optional[List[torch.Tensor]] = None


class ComputeEmbeddingsWrapper:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Callable,
        tokenizer_options: Dict = {},
        post_processing_function: Optional[Callable] = None,
        device: torch.device = None,
        pad_token_id: int = 0,
        low_memory: bool = False,
        save_directories: Optional[SaveDirectories] = None,
    ):
        if low_memory and save_directories is None:
            raise ValueError(
                "Expected `save_path` to have a value given that "
                "`low_memory=True`. "
                f"Received: save_path={save_directories}."
            )

        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_options = tokenizer_options
        self.post_processing_function = post_processing_function
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device = device
        self.pad_token_id = pad_token_id
        self.low_memory = low_memory
        self.save_directories = save_directories

    def __call__(
        self, train_seqs, val_seqs=None, test_seqs=None
    ) -> EmbeddingsContainer:
        if self.low_memory:
            compute_embeddings_and_save_to_disk(
                self.model,
                self.tokenizer,
                self.save_directories,
                train_seqs=train_seqs,
                val_seqs=val_seqs,
                test_seqs=test_seqs,
                post_processing_fn=self.post_processing_function,
                pad_token_id=self.pad_token_id,
                tokenizer_options=self.tokenizer_options,
            )
        else:
            outputs = compute_embeddings(
                self.model,
                self.tokenizer,
                train_seqs=train_seqs,
                val_seqs=val_seqs,
                test_seqs=test_seqs,
                post_processing_fn=self.post_processing_function,
                device=self.device,
                pad_token_id=self.pad_token_id,
                tokenizer_options=self.tokenizer_options,
            )
            train_embeds, val_embeds, test_embeds = outputs
            return EmbeddingsContainer(
                train_embeddings=train_embeds,
                val_embeddings=val_embeds,
                test_embeddings=test_embeds,
            )
