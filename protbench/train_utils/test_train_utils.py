import unittest
import random

import torch

from protbench.src.train_utils.train_utils import (
    collate_inputs,
    collate_inputs_and_labels,
    preprocess_binary_classification_logits,
    preprocess_multi_classification_logits,
)

torch.manual_seed(42)
random.seed(42)


class TestCollators(unittest.TestCase):
    def test_collate_inputs(self):
        seq_min_len = 80
        seq_max_len = 120
        batch_size = 3
        embedding_dim = 16
        padding_value = 0

        inputs = [
            {
                "embd": torch.rand(seq_max_len, embedding_dim),
                "labels": torch.rand(1).squeeze(),
            },
            {
                "embd": torch.rand(seq_min_len, embedding_dim),
                "labels": torch.rand(1).squeeze(),
            },
            {
                "embd": torch.rand(
                    random.randint(seq_min_len, seq_max_len), embedding_dim
                ),
                "labels": torch.rand(1).squeeze(),
            },
        ]
        collated_inputs = collate_inputs(inputs, padding_value=padding_value)
        self.assertEqual(
            collated_inputs["embd"].shape,
            (batch_size, seq_max_len, embedding_dim),
        )
        self.assertEqual(collated_inputs["labels"].shape, (batch_size,))

        for i, (embd, label) in enumerate(
            zip(collated_inputs["embd"], collated_inputs["labels"])
        ):
            original_seq_len = inputs[i]["embd"].size(0)
            self.assertTrue(torch.equal(label, inputs[i]["labels"]))
            self.assertTrue(
                torch.equal(inputs[i]["embd"], embd[:original_seq_len])
            )
            torch.equal(
                embd[original_seq_len:],
                torch.zeros(
                    (seq_max_len - inputs[i]["embd"].size(0), embedding_dim)
                ),
            )

    def test_collate_inputs_and_labels(self):
        seq_min_len = 80
        seq_max_len = 120
        batch_size = 3
        embedding_dim = 16
        padding_value = 0
        label_padding_value = -20

        inputs = [
            {
                "embd": torch.rand(seq_max_len, embedding_dim),
                "labels": torch.rand(seq_max_len),
            },
            {
                "embd": torch.rand(seq_min_len, embedding_dim),
                "labels": torch.rand(seq_min_len),
            },
            {
                "embd": torch.rand(
                    (seq_max_len + seq_min_len) // 2, embedding_dim
                ),
                "labels": torch.rand(seq_max_len),
            },
        ]
        collated_inputs = collate_inputs_and_labels(
            inputs,
            input_padding_value=padding_value,
            label_padding_value=label_padding_value,
        )
        self.assertEqual(
            collated_inputs["embd"].shape,
            (batch_size, seq_max_len, embedding_dim),
        )
        self.assertEqual(
            collated_inputs["labels"].shape, (batch_size, seq_max_len)
        )

        for i, (embd, label) in enumerate(
            zip(collated_inputs["embd"], collated_inputs["labels"])
        ):
            original_seq_len = inputs[i]["embd"].size(0)
            original_label_len = inputs[i]["labels"].size(0)
            self.assertTrue(
                torch.equal(inputs[i]["labels"], label[:original_label_len])
            )
            torch.equal(
                label[original_label_len:],
                torch.zeros(
                    (seq_max_len - inputs[i]["embd"].size(0), embedding_dim)
                )
                + label_padding_value,
            )
            self.assertTrue(
                torch.equal(inputs[i]["embd"], embd[:original_seq_len])
            )
            torch.equal(
                embd[original_seq_len:],
                torch.zeros(
                    (seq_max_len - inputs[i]["embd"].size(0), embedding_dim)
                ),
            )


class TestPreprocessing(unittest.TestCase):
    def test_preprocess_multi_classification_logits(self):
        logits = torch.tensor(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
                [[1, 3, 2], [6, 5, 4], [7, 8, 9]],
            ]
        )
        labels = torch.rand(1)
        expected_output = torch.tensor([[2, 2, 2], [0, 0, 0], [1, 0, 2]])
        self.assertTrue(
            torch.equal(
                preprocess_multi_classification_logits(logits, labels),
                expected_output,
            )
        )

    def test_preprocess_binary_classification_logits(self):
        logits = torch.tensor([[-0.1], [-0.3], [0.6], [0.9]])
        labels = torch.rand(1)
        expected_output = torch.tensor([[0], [0], [1], [1]])
        self.assertTrue(
            torch.equal(
                preprocess_binary_classification_logits(logits, labels),
                expected_output,
            )
        )
