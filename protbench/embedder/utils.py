from __future__ import annotations

import os
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple

import torch

from protbench.embedder import TorchEmbedder
from protbench.embedder import TorchEmbeddingFunction


def delete_directory_contents(directory: PathLike):
    if not Path(directory).exists():
        return
    for root, _, files in os.walk(directory, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))


@dataclass
class SaveDirectories:
    parent_dir: PathLike = "/root/.cache"
    train_dir: PathLike = "train_embeddings"
    validation_dir: PathLike = "validation_embeddings"
    test_dir: PathLike = "test_embeddings"

    def create_or_clean_directory(self, directory: PathLike):
        delete_directory_contents(directory=directory)
        Path(directory).mkdir(exist_ok=True)

    @property
    def train(self) -> PathLike:
        return os.path.join(self.parent_dir, self.train_dir)

    @property
    def validation(self) -> PathLike:
        return os.path.join(self.parent_dir, self.validation_dir)

    @property
    def test(self) -> PathLike:
        return os.path.join(self.parent_dir, self.test_dir)


def compute_embeddings(
    model: torch.nn.Module,
    tokenization_fn: Callable,
    train_seqs: List[str],
    val_seqs: List[str] | None = None,
    test_seqs: List[str] | None = None,
    forward_options: Dict | None = {},
    post_processing_fn: Callable = None,
    device: torch.device | None = None,
    pad_token_id: int = 0,
):
    embedding_fn = TorchEmbeddingFunction(
        model=model,
        tokenization_fn=tokenization_fn,
        device=device,
        forward_options=forward_options,
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

    for data in [train_seqs, val_seqs, test_seqs]:
        if data is not None:
            embeddings.append(embedder.run(data))
        else:
            embeddings.append(None)

    return embeddings


def compute_embeddings_and_save_to_disk(
    model: torch.nn.Module,
    tokenization_fn: Callable,
    save_directories: SaveDirectories,
    train_seqs: List[str],
    val_seqs: List[str] | None = None,
    test_seqs: List[str] | None = None,
    forward_options: Dict | None = {},
    post_processing_fn: Callable = None,
    device: torch.device | None = None,
    pad_token_id: int = 0,
):
    embedding_fn = TorchEmbeddingFunction(
        model=model,
        tokenization_fn=tokenization_fn,
        device=device,
        forward_options=forward_options,
        embeddings_postprocessing_fn=post_processing_fn,
        pad_token_id=pad_token_id,
    )

    save_directories.create_or_clean_directory(save_directories.train)

    compute_data = [
        (train_seqs, save_directories.train),
    ]

    if val_seqs is not None:
        save_directories.create_or_clean_directory(save_directories.validation)
        compute_data.append((val_seqs, save_directories.validation))

    if test_seqs is not None:
        save_directories.create_or_clean_directory(save_directories.test)
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


class EmbeddingsContainer(NamedTuple):
    train_embeddings: List[torch.Tensor]
    val_embeddings: List[torch.Tensor] | None = None
    test_embeddings: List[torch.Tensor] | None = None


class ComputeEmbeddingsWrapper:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenization_fn: Callable,
        forward_options: Dict = {},
        post_processing_function: Callable | None = None,
        device: torch.device = None,
        pad_token_id: int = 0,
        low_memory: bool = False,
        save_directories: SaveDirectories | None = None,
    ):
        if low_memory and save_directories is None:
            raise ValueError(
                "Expected `save_path` to have a value given that "
                "`low_memory=True`. "
                f"Received: save_path={save_directories}."
            )

        self.model = model
        self.tokenization_fn = tokenization_fn
        self.post_processing_function = post_processing_function
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device = device
        self.pad_token_id = pad_token_id
        self.low_memory = low_memory
        self.save_directories = save_directories
        self.forward_options = forward_options

    def __call__(
        self, train_seqs, val_seqs=None, test_seqs=None
    ) -> EmbeddingsContainer | None:
        if self.low_memory:
            compute_embeddings_and_save_to_disk(
                model=self.model,
                tokenization_fn=self.tokenization_fn,
                save_directories=self.save_directories,
                train_seqs=train_seqs,
                val_seqs=val_seqs,
                test_seqs=test_seqs,
                forward_options=self.forward_options,
                post_processing_fn=self.post_processing_function,
                pad_token_id=self.pad_token_id,
            )
        else:
            outputs = compute_embeddings(
                model=self.model,
                tokenization_fn=self.tokenization_fn,
                train_seqs=train_seqs,
                val_seqs=val_seqs,
                test_seqs=test_seqs,
                forward_options=self.forward_options,
                post_processing_fn=self.post_processing_function,
                device=self.device,
                pad_token_id=self.pad_token_id,
            )
            train_embeds, val_embeds, test_embeds = outputs
            return EmbeddingsContainer(
                train_embeddings=train_embeds,
                val_embeddings=val_embeds,
                test_embeddings=test_embeds,
            )
