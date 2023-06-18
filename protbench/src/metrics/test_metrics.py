import unittest

import numpy as np
from transformers import EvalPrediction

from protbench.src.metrics import MetricRegistry
from protbench.src.metrics.metrics import (
    compute_accuracy,
)


class TestMetricRegistry(unittest.TestCase):
    def test_add_metric(self):
        if "test_metric" in MetricRegistry.metric_name_map:
            del MetricRegistry.metric_name_map["test_metric"]

        @MetricRegistry.register("test_metric")
        def test_metric():
            return "test"

        self.assertIn("test_metric", MetricRegistry.metric_name_map)
        self.assertIs(MetricRegistry.metric_name_map["test_metric"], test_metric)
        self.assertEqual(MetricRegistry.metric_name_map["test_metric"], test_metric)

    def test_add_metric_with_existing_name(self):
        if "test_metric" in MetricRegistry.metric_name_map:
            del MetricRegistry.metric_name_map["test_metric"]

        @MetricRegistry.register("test_metric")
        def test_metric():
            return "test"

        with self.assertRaises(ValueError):

            @MetricRegistry.register("test_metric")
            def test_metric2():
                return "test2"


class TestMetrics(unittest.TestCase):
    def test_compute_binary_classification_accuracy(self):
        p = EvalPrediction(
            predictions=np.array([0, 1, 0, 1]), label_ids=np.array([0, 1, 0, 1])
        )
        self.assertEqual(compute_accuracy(p), 1.0)
        p = EvalPrediction(
            predictions=np.array([0, 1, 0, 1]), label_ids=np.array([0, 1, 1, 0])
        )
        self.assertEqual(compute_accuracy(p), 0.5)
        p = EvalPrediction(
            predictions=np.array([0, 1, 0, 1]), label_ids=np.array([1, 0, 1, 0])
        )
        self.assertEqual(compute_accuracy(p), 0.0)

    def test_multiclass_classification_accuracy(self):
        p = EvalPrediction(
            predictions=np.array([0, 1, 2, 0, 1, 2]),
            label_ids=np.array([0, 1, 2, 0, 1, 2]),
        )
        self.assertEqual(compute_accuracy(p), 1.0)
        p = EvalPrediction(
            predictions=np.array([0, 1, 2, 2, 0, 0]),
            label_ids=np.array([0, 1, 2, 0, 1, 2]),
        )
        self.assertEqual(compute_accuracy(p), 0.5)
        p = EvalPrediction(
            predictions=np.array([0, 0, 2, 0, 2, 2]),
            label_ids=np.array([2, 1, 0, 2, 1, 0]),
        )
        self.assertEqual(compute_accuracy(p), 0.0)

    def test_compute_accuracy_with_ignore_index(self):
        ignore_index = -100
        p = EvalPrediction(
            predictions=np.array([0, 1, 0, 0, 1, 0]),
            label_ids=np.array([0, 1, ignore_index, 0, 1, ignore_index]),
        )
        self.assertEqual(compute_accuracy(p, ignore_index=ignore_index), 1.0)
        ignore_index = 30
        p = EvalPrediction(
            predictions=np.array([0, 1, 0, 0, 1, 0]),
            label_ids=np.array([0, 1, ignore_index, 1, 0, ignore_index]),
        )
        self.assertEqual(compute_accuracy(p, ignore_index=ignore_index), 0.5)
