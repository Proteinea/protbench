import unittest

import torch

from protbench.src.models.downstream import convbert_layers
from protbench.src.models.downstream.downstream_models import (
    ConvBertForMultiClassTokenClassification,
    ConvBertForBinaryTokenClassification,
    ConvBertForMultiClassSeqClassification,
    ConvBertForBinarySeqClassification,
    ConvBertForRegression,
)

torch.manual_seed(42)


class TestBaseConvBertLayers(unittest.TestCase):
    @torch.inference_mode()
    def test_base_convbert_with_attention_mask(self):
        embedding_dim = 32
        nhead = 1
        hidden_dim = int(embedding_dim / 2)
        num_layers = 1
        kernel_size = 3
        convbert_encoder = convbert_layers.BaseConvBert(
            input_dim=embedding_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=0.0,
        ).eval()

        batch_size = 4
        seq_len = 100
        embds = torch.rand(batch_size, seq_len, embedding_dim)

        output_without_padding = convbert_encoder.convbert_forward(embds)[0]
        self.assertEqual(
            output_without_padding.shape, (batch_size, seq_len, embedding_dim)
        )

        # test with padding
        padding_len = 5
        padded_embds = torch.cat(
            (embds, torch.zeros(batch_size, padding_len, embedding_dim)), dim=1
        )
        output_with_padding = convbert_encoder.convbert_forward(padded_embds)[0]
        self.assertEqual(
            output_with_padding.shape,
            (batch_size, seq_len + padding_len, embedding_dim),
        )

        self.assertTrue(
            torch.equal(output_with_padding[:, :seq_len, :], output_without_padding)
        )


class TestConvBertForTokenClassification(unittest.TestCase):
    @torch.inference_mode()
    def test_multiclass_token_classification(self):
        embedding_dim = 32
        nhead = 1
        hidden_dim = int(embedding_dim / 2)
        num_layers = 1
        kernel_size = 3
        num_tokens = 3
        loss_ignore_index = -50
        convbert_model = ConvBertForMultiClassTokenClassification(
            input_dim=embedding_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=0.0,
            num_tokens=num_tokens,
            loss_ignore_index=loss_ignore_index,
        ).eval()

        batch_size = 4
        seq_len = 100
        embds = torch.rand(batch_size, seq_len, embedding_dim)
        labels = torch.randint(0, num_tokens, (batch_size, seq_len))

        output_without_padding = convbert_model(embds, labels=labels)
        self.assertEqual(
            output_without_padding.logits.shape, (batch_size, seq_len, num_tokens)
        )

        # test with padding
        padding_len = 5
        padded_embds = torch.cat(
            (embds, torch.zeros(batch_size, padding_len, embedding_dim)), dim=1
        )
        padded_labels = torch.cat(
            (labels, torch.zeros(batch_size, padding_len) + loss_ignore_index), dim=1
        ).long()
        output_with_padding = convbert_model(padded_embds, labels=padded_labels)
        self.assertEqual(
            output_with_padding.logits.shape,
            (batch_size, seq_len + padding_len, num_tokens),
        )
        self.assertTrue(
            torch.equal(
                output_with_padding.logits[:, :seq_len, :],
                output_without_padding.logits,
            )
        )
        self.assertTrue(
            torch.isclose(output_with_padding.loss, output_without_padding.loss)
        )

    @torch.inference_mode()
    def test_binary_token_classification(self):
        embedding_dim = 32
        nhead = 1
        hidden_dim = int(embedding_dim / 2)
        num_layers = 1
        kernel_size = 3
        loss_ignore_index = -80
        convbert_model = ConvBertForBinaryTokenClassification(
            input_dim=embedding_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=0.0,
            loss_ignore_index=loss_ignore_index,
        ).eval()

        batch_size = 4
        seq_len = 100
        embds = torch.rand(batch_size, seq_len, embedding_dim)
        labels = torch.randint(0, 2, (batch_size, seq_len))

        output_without_padding = convbert_model(embds, labels=labels)
        self.assertEqual(output_without_padding.logits.shape, (batch_size, seq_len, 1))

        # test with padding
        padding_len = 5
        padded_embds = torch.cat(
            (embds, torch.zeros(batch_size, padding_len, embedding_dim)), dim=1
        )
        padded_labels = torch.cat(
            (labels, torch.zeros(batch_size, padding_len) + loss_ignore_index), dim=1
        ).long()
        output_with_padding = convbert_model(padded_embds, labels=padded_labels)
        self.assertEqual(
            output_with_padding.logits.shape,
            (batch_size, seq_len + padding_len, 1),
        )
        self.assertTrue(
            torch.equal(
                output_with_padding.logits[:, :seq_len, :],
                output_without_padding.logits,
            )
        )
        self.assertTrue(
            torch.isclose(output_with_padding.loss, output_without_padding.loss)
        )


class TestConvBertForSeqClassification(unittest.TestCase):
    @torch.inference_mode()
    def test_multiclass_seq_classification(self):
        embedding_dim = 32
        nhead = 1
        hidden_dim = int(embedding_dim / 2)
        num_layers = 1
        kernel_size = 3
        num_tokens = 3
        pooling = "max"
        convbert_model = ConvBertForMultiClassSeqClassification(
            input_dim=embedding_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=0.0,
            num_tokens=num_tokens,
            pooling=pooling,
        ).eval()

        batch_size = 4
        seq_len = 100
        embds = torch.rand(batch_size, seq_len, embedding_dim)
        labels = torch.randint(0, num_tokens, (batch_size,))

        output_without_padding = convbert_model(embds, labels=labels)
        self.assertEqual(output_without_padding.logits.shape, (batch_size, num_tokens))

        # test with padding
        padding_len = 5
        padded_embds = torch.cat(
            (embds, torch.zeros(batch_size, padding_len, embedding_dim)), dim=1
        )
        output_with_padding = convbert_model(padded_embds, labels=labels)
        self.assertEqual(
            output_with_padding.logits.shape,
            (batch_size, num_tokens),
        )
        self.assertTrue(
            torch.equal(output_with_padding.logits, output_without_padding.logits)
        )
        self.assertTrue(
            torch.equal(output_with_padding.loss, output_without_padding.loss)
        )

    @torch.inference_mode()
    def test_binary_seq_classification(self):
        embedding_dim = 32
        nhead = 1
        hidden_dim = int(embedding_dim / 2)
        num_layers = 1
        kernel_size = 3
        pooling = "mean"
        convbert_model = ConvBertForBinarySeqClassification(
            input_dim=embedding_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=0.0,
            pooling=pooling,
        ).eval()

        batch_size = 4
        seq_len = 100
        embds = torch.rand(batch_size, seq_len, embedding_dim)
        labels = torch.randint(0, 2, (batch_size,))

        output_without_padding = convbert_model(embds, labels=labels)
        self.assertEqual(output_without_padding.logits.shape, (batch_size, 1))

        # test with padding
        padding_len = 5
        padded_embds = torch.cat(
            (embds, torch.zeros(batch_size, padding_len, embedding_dim)), dim=1
        )
        output_with_padding = convbert_model(padded_embds, labels=labels)
        self.assertEqual(
            output_with_padding.logits.shape,
            (batch_size, 1),
        )
        self.assertTrue(
            torch.equal(output_with_padding.logits, output_without_padding.logits)
        )
        self.assertTrue(
            torch.equal(output_with_padding.loss, output_without_padding.loss)
        )


class TestConvBertForRegression(unittest.TestCase):
    @torch.inference_mode()
    def test_regression(self):
        embedding_dim = 32
        nhead = 1
        hidden_dim = int(embedding_dim / 2)
        num_layers = 1
        kernel_size = 3
        pooling = "max"
        convbert_model = ConvBertForRegression(
            input_dim=embedding_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=0.0,
            pooling=pooling,
        ).eval()

        batch_size = 4
        seq_len = 100
        embds = torch.rand(batch_size, seq_len, embedding_dim)
        labels = torch.rand(batch_size, 1)

        output_without_padding = convbert_model(embds, labels=labels)
        self.assertEqual(output_without_padding.logits.shape, (batch_size, 1))

        # test with padding
        padding_len = 5
        padded_embds = torch.cat(
            (embds, torch.zeros(batch_size, padding_len, embedding_dim)), dim=1
        )
        output_with_padding = convbert_model(padded_embds, labels=labels)
        self.assertEqual(
            output_with_padding.logits.shape,
            (batch_size, 1),
        )
        self.assertTrue(
            torch.equal(output_with_padding.logits, output_without_padding.logits)
        )
        self.assertTrue(
            torch.equal(output_with_padding.loss, output_without_padding.loss)
        )
