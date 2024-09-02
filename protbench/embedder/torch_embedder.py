from __future__ import annotations
import logging
import math
from math import ceil
from os import PathLike
from typing import Iterable
from typing import List

import torch
from tqdm.auto import tqdm

from protbench.embedder.embedder import Embedder
from protbench.embedder.torch_embedding_function import TorchEmbeddingFunction


class TorchEmbedder(Embedder):
    def __init__(
        self,
        embedding_function: TorchEmbeddingFunction,
        low_memory: bool = False,
        save_path: PathLike | None = None,
        devices: List[torch.device] | List[int] | None = None,
        batch_size: int = 1,
    ):
        """Embedder for torch models. Can be used to embed sequences in batches
           and on multiple GPUs.

        Args:
            embedding_function (TorchEmbeddingFunction): PyTorch embedding
                                                         function.
            low_memory (bool, optional): Reduce memory usage by not storing
                                         embeddings in memory.
                                         Defaults to False.
            save_path (PathLike | None, optional): Path to save embeddings to.
                Defaults to None.
                Embeddings are saved as numpy arrays with the name of the file
                being the index of the embedding.
                For example, the embedding of the first sequence will be saved
                as save_path/0.npy.
            devices (List[torch.device] | List[int] | None, optional):
                List of devices to use for embedding.
                If None, all available cuda devices will be used.
                Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 1.
        """
        super(TorchEmbedder, self).__init__(
            embedding_function, low_memory, save_path
        )
        if devices is None:
            devices = [
                torch.device(f"cuda:{i}")
                for i in range(torch.cuda.device_count())
            ]

        if len(devices) == 0:
            devices = [torch.device("cpu")]

        self.devices = devices
        self.batch_size = batch_size

    @staticmethod
    def _get_sequences_for_current_process(
        rank: int, world_size: int, sequences: Iterable[str]
    ) -> Iterable[str]:
        """Helper function to get the sequences for the current process in
            a parallel embedding run. This function is used to split the
            sequences into world_size batches for each process.

        Args:
            rank (int): process rank or index.
            world_size (int): total number of processes.
            sequences (Iterable[str]): sequences to embed.

        Returns:
            Iterable[str]: sequences for the current process.
        """
        num_sequences_per_device = math.ceil(len(sequences) / world_size)
        batch_start = rank * num_sequences_per_device
        batch_end = batch_start + num_sequences_per_device
        sequences_per_device = sequences[batch_start:batch_end]
        return sequences_per_device, batch_start

    def _collect_embeddings_from_processes(
        self,
        output_queue: torch.multiprocessing.Queue,
        total_sequences: int,
    ) -> List[torch.Tensor]:
        """Collect embeddings from processes. This function is used in a
           parallel embedding run to collect the embeddings sent by the child
           processes through the output_queue.

        Args:
            output_queue (torch.multiprocessing.Queue): Shared queue to collect
                                                        embeddings from.
            total_sequences (int): Total number of sequences to embed.

        Returns:
            List[torch.Tensor]: List of PyTorch Tensors.
        """
        embeddings = [None] * total_sequences
        with tqdm(
            desc="Embedding sequences: ",
            unit=" sequence",
            total=total_sequences,
            colour="blue",
        ) as pbar:
            for _ in range(total_sequences):
                i, embedding = output_queue.get()
                assert embeddings[i] is None
                embeddings[i] = torch.from_numpy(embedding)
                pbar.update(1)
            assert output_queue.empty()
        return embeddings

    def _run_single_process(self, sequences: Iterable[str]) -> List[torch.Tensor]:
        # run on a single gpu
        num_batches = ceil(len(sequences) / self.batch_size)
        embeddings = []
        self.embedding_function.device = self.devices[0]
        self.embedding_function.model.to(self.devices[0])
        for batch_start in tqdm(
            range(0, len(sequences), self.batch_size),
            total=num_batches,
            desc="Embedding sequences: ",
            unit=" batch",
            leave=False,
        ):
            batch_end = batch_start + self.batch_size
            batch_sequences = sequences[batch_start:batch_end]
            batch_embeddings = self.embedding_function(
                batch_sequences, remove_padding=True
            )
            for i, embedding in enumerate(batch_embeddings, start=batch_start):
                if self.save_path is not None:
                    with torch.multiprocessing.Lock():
                        TorchEmbedder.save_embedding_to_disk(
                            i, embedding.numpy(), self.save_path
                        )
                if not self.low_memory:
                    embeddings.append(embedding)
        return embeddings

    def _run_multiple_processes(
        self, sequences: Iterable[str]
    ) -> List[torch.Tensor]:
        # run on multiple gpus
        output_queue = (
            torch.multiprocessing.Queue(maxsize=len(sequences))
            if not self.low_memory
            else None
        )
        logging.info(
            f"Using {', '.join([str(d) for d in self.devices])} devices to "
            "embed sequences..."
        )
        processes = []
        for i in range(len(self.devices)):
            p = torch.multiprocessing.Process(
                target=self._run_entrypoint,
                args=(
                    i,
                    self.devices,
                    len(self.devices),
                    sequences,
                    self.batch_size,
                    self.embedding_function,
                    output_queue,
                    self.save_path,
                ),
            )
            p.start()
            processes.append(p)
        if not self.low_memory:
            embeddings = self._collect_embeddings_from_processes(
                output_queue, len(sequences)
            )
        else:
            embeddings = []
        for p in processes:
            p.join()
        return embeddings

    def run(self, sequences: Iterable[str]) -> List[torch.Tensor]:
        """Run the embedding function on the given sequences.

        Args:
            sequences (Iterable[str]): sequences to embed.

        Returns:
            List[torch.Tensor]: list of embeddings. If low_memory is True,
                this list will be empty.
        """
        if len(self.devices) == 1:
            return self._run_single_process(sequences)
        else:
            return self._run_multiple_processes(sequences)

    @staticmethod
    def _run_entrypoint(
        rank: int,
        devices,
        world_size: int,
        sequences: Iterable[str],
        batch_size: int,
        embedding_function: TorchEmbeddingFunction,
        output_queue: torch.multiprocessing.Queue | None = None,
        save_path: PathLike | None = None,
    ):
        # Entry point for parallel embedding run. This function is used to run
        # the embedding function on a single gpu in a parallel embedding run.
        # Notes:
        # - This function is not meant to be called directly. It is called by
        # the _run_multiple_gpus function.
        # - Why does this function send the embeddings to the parent process
        #   as a numpy array?
        #   This is because for some reason the torch tensors are not properly
        #   garbage collected after
        #   the parent process receives them. This causes shared memory leaks
        #   which eventually leads to an
        #   error with the process hitting the maximum number of open shared
        #   files. This is a
        #   known issue with torch multiprocessing. More details can be found
        #   here: https://github.com/pytorch/pytorch/issues/973.
        #   Also, assuming the issue is fixed, it is probably still better to
        #   send the embeddings as numpy arrays
        #   since they lead to much faster pickling and unpickling (in my use
        #   case, the embeddings are 2x faster to send).

        device = devices[rank]
        embedding_function.device = device
        embedding_function.model.to(device)
        (
            sequences,
            start_idx,
        ) = TorchEmbedder._get_sequences_for_current_process(
            rank, world_size, sequences
        )
        for batch_start in range(0, len(sequences), batch_size):
            batch_end = batch_start + batch_size
            batch_sequences = sequences[batch_start:batch_end]
            batch_embeddings = embedding_function(
                batch_sequences, remove_padding=True
            )
            for i, embedding in enumerate(
                batch_embeddings, start=batch_start + start_idx
            ):
                if save_path is not None:
                    TorchEmbedder.save_embedding_to_disk(
                        i, embedding.numpy(), save_path
                    )
                if output_queue is not None:
                    output_queue.put((i, embedding.numpy()))
