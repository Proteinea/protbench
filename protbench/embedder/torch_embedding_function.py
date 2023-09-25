from typing import List, Callable, Any, Union

import torch

from protbench.embedder import EmbeddingFunction


class TorchEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Callable[[List[str]], torch.Tensor],
        device: Union[int, torch.device],
        embeddings_postprocessing_fn: Callable[[Any], torch.Tensor] = None,
        pad_token_id: int = 0,
    ):
        """Embedding function for pytorch models.

        Args:
            model (torch.nn.Module): pytorch module.
            tokenizer (Callable[[List[str]], torch.Tensor]): tokenizer.
            device (Union[int, torch.device]): device to use for embedding.
            embeddings_postprocessing_fn (Callable[[Any], torch.Tensor], optional):
                function to apply to the model outputs before returning them. Defaults to None.
                This can be used to extract the last hidden state from a HF ModelOutput object.
            pad_token_id (int, optional): tokenizer's pad token id. Defaults to 0.
        """
        super().__init__(model, tokenizer)
        if self.model.training:
            self.model.eval()
        self.embeddings_postprocessing_fn = embeddings_postprocessing_fn
        self.device = device
        self.pad_token_id = pad_token_id

    @staticmethod
    def _remove_padding_from_embeddings(
        embeddings: torch.Tensor, input_ids: torch.Tensor, padding_value: int = 0
    ) -> List[torch.Tensor]:
        """Remove padding from embeddings.

        Args:
            embeddings (torch.Tensor): embeddings of shape (batch_size, seq_len, embd_dim).
            input_ids (torch.Tensor): input_ids of shape (batch_size, seq_len).
            padding_value (int, optional): padding value in input ids. Defaults to 0.

        Returns:
            List[torch.Tensor]: list of tensors of embeddings without padding.
        """
        embeddings_without_padding = []
        for i in range(embeddings.shape[0]):
            seq_len = (input_ids[i, :] != padding_value).sum()
            embeddings_without_padding.append(embeddings[i, :seq_len, :])
        return embeddings_without_padding

    @torch.inference_mode()
    def call(
        self, sequences: List[str], remove_padding: bool = True
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Embed a list of sequences.

        Args:
            sequences (List[str]): list of sequences to embed.
            remove_padding (bool, optional): remove padding from embeddings. Defaults to True.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: if remove_padding is True, returns a list of
                tensors of embeddings without padding where each tensor has shape (seq_len, embd_dim).
                If remove_padding is False, returns a tensor of embeddings with shape
                (batch_size, max_seq_len, embd_dim).
        """
        input_ids = self.tokenizer(sequences)
        try:
            model_outputs = self.model(input_ids.to(self.device))
        except torch.cuda.OutOfMemoryError:
            self.model.cpu()
            model_outputs = self.model(input_ids.cpu())
            self.model.to(self.device)
        if self.embeddings_postprocessing_fn is not None:
            model_outputs = self.embeddings_postprocessing_fn(model_outputs)
        model_outputs = model_outputs.to("cpu")
        if remove_padding:
            model_outputs = self._remove_padding_from_embeddings(
                model_outputs, input_ids, self.pad_token_id
            )
        return model_outputs
