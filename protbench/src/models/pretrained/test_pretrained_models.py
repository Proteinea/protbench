import unittest
import random

import torch
from transformers import AutoModel, AutoTokenizer

from protbench.src.models.pretrained.pretrained_models import T5BasedModels

random.seed(42)


class TestHuggingfaceModels(unittest.TestCase):
    def generate_random_sequence(self, length: int) -> str:
        return "".join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=length))

    @torch.inference_mode()
    def embed_sequence(self, sequence: str, model, tokenizer) -> torch.Tensor:
        batch_encoding = tokenizer(
            [sequence],
            padding="do_not_pad",
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
        )

        model.eval()
        output = model(**batch_encoding)
        return output.last_hidden_state.squeeze(0)

    def test_embed_sequences(self):
        test_model = "ElnaggarLab/ankh-base"  # using ankh base as the test model
        tokenizer = AutoTokenizer.from_pretrained(test_model)
        model = AutoModel.from_pretrained(test_model).get_encoder()

        num_sequences = 10
        seq_min_length = 80
        seq_max_length = 120
        sequences = [
            self.generate_random_sequence(
                length=random.randint(seq_min_length, seq_max_length)
            )
            for _ in range(num_sequences)
        ]

        ground_truth_embeddings = [
            self.embed_sequence(seq, model, tokenizer) for seq in sequences
        ]

        batch_size = 1
        num_workers = 0
        hf_models = T5BasedModels(
            test_model, batch_size=batch_size, num_workers=num_workers
        )
        embeddings = hf_models.embed_sequences(sequences, torch.device("cpu"))

        for embd, gt_embd in zip(embeddings, ground_truth_embeddings):
            self.assertTrue(torch.equal(embd, gt_embd))

        # test batch size > 1 to ensure that final embeddings are not padded
        batch_size = 4
        num_workers = 0
        hf_models = T5BasedModels(
            test_model, batch_size=batch_size, num_workers=num_workers
        )
        embeddings = hf_models.embed_sequences(sequences, torch.device("cpu"))

        for embd, gt_embd in zip(embeddings, ground_truth_embeddings):
            self.assertTrue(torch.equal(embd, gt_embd))

        # test num_workers > 0
        batch_size = 4
        num_workers = 1
        hf_models = T5BasedModels(
            test_model, batch_size=batch_size, num_workers=num_workers
        )
        embeddings = hf_models.embed_sequences(sequences, torch.device("cpu"))

        for embd, gt_embd in zip(embeddings, ground_truth_embeddings):
            self.assertTrue(torch.equal(embd, gt_embd))
