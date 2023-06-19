import unittest

from protbench.src.models import DownstreamModelRegistry, PretrainedModelRegistry


class TestPretrainedModelRegistry(unittest.TestCase):
    def test_add_pretrained_model(self):
        if "test_model" in PretrainedModelRegistry.pretrained_model_name_map:
            del PretrainedModelRegistry.pretrained_model_name_map["test_model"]

        @PretrainedModelRegistry.register("test_model")
        class TestModel:
            pass

        self.assertIn("test_model", PretrainedModelRegistry.pretrained_model_name_map)
        self.assertEqual(
            PretrainedModelRegistry.pretrained_model_name_map["test_model"], TestModel
        )

    def test_add_pretrained_model_with_existing_name(self):
        if "test_model" in PretrainedModelRegistry.pretrained_model_name_map:
            del PretrainedModelRegistry.pretrained_model_name_map["test_model"]

        @PretrainedModelRegistry.register("test_model")
        class TestModel:
            pass

        with self.assertRaises(ValueError):

            @PretrainedModelRegistry.register("test_model")
            class TestModel2:
                pass

class TestDownstreamModelRegistry(unittest.TestCase):
    def test_add_downstream_model(self):
        if "test_model" in DownstreamModelRegistry.downstream_model_name_map:
            del DownstreamModelRegistry.downstream_model_name_map["test_model"]

        @DownstreamModelRegistry.register("test_model")
        class TestModel:
            pass

        self.assertIn("test_model", DownstreamModelRegistry.downstream_model_name_map)
        self.assertEqual(
            DownstreamModelRegistry.downstream_model_name_map["test_model"], TestModel
        )

    def test_add_downstream_model_with_existing_name(self):
        if "test_model" in DownstreamModelRegistry.downstream_model_name_map:
            del DownstreamModelRegistry.downstream_model_name_map["test_model"]

        @DownstreamModelRegistry.register("test_model")
        class TestModel:
            pass

        with self.assertRaises(ValueError):

            @DownstreamModelRegistry.register("test_model")
            class TestModel2:
                pass
