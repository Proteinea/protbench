import logging
from typing import Dict, Optional, Callable, Type, Any

import torch


class DownstreamModelRegistry:
    """Central repository for all models.

    - To register a new downstream model you can use the `register` decorator as follows:
    @ModelRegistry.register('my_new_model')
    class MyNewModel(torch.nn.Module):
    ...

    - Or you can use the `register` method directly:

    class MyNewModel(torch.nn.Module):
    ...
    ModelRegistry.register('my_new_model', MyNewModel)
    """

    downstream_model_name_map: Dict[str, Type[torch.nn.Module]] = {}

    @classmethod
    def register(
        cls,
        model_name: str,
        model_cls: Optional[Type[torch.nn.Module]] = None,
    ) -> Callable | Type[torch.nn.Module]:
        """Register a new downstream model. This can be used as a decorator providing only the model_name
        or directly as a method providing the model_name and model class.

        Args:
            model_name (str): name of the model. Must be unique. Same name will be used in
                the config file to refer to the model.
            model_cls (Optional[Type[torch.nn.Module]], optional): model class. Defaults to None.
        """
        if model_name in cls.downstream_model_name_map:
            raise ValueError(
                f"Downstream model {model_name} already exists in the registry. "
                f"Please choose a different name."
            )
        if model_cls is None:  # expected when using decorator
            return lambda model_cls: cls.register(model_name, model_cls)
        if not issubclass(model_cls, torch.nn.Module):
            logging.warning(
                f"Downstream model {model_name} does not inherit from the torch.nn.Module."
            )
        cls.downstream_model_name_map[model_name] = model_cls
        return model_cls


class PretrainedModelRegistry:
    """Central repository for pretrained models.

    - To register a new pretrained model you can use the `register` decorator as follows:
    @PretrainedModelRegistry.register('my_new_model')
    class MyNewModel(torch.nn.Module):
    ...

    - Or you can use the `register` method directly:

    class MyNewModel(torch.nn.Module):
    ...
    PretrainedModelRegistry.register('my_new_model', MyNewModel)

    """

    pretrained_model_name_map: Dict[str, Type[torch.nn.Module]] = {}

    @classmethod
    def register(
        cls,
        model_name: str,
        model_cls: Any = None,
    ) -> Callable | Type[torch.nn.Module]:
        """Register a new pretrained model. This can be used as a decorator providing only the model_name
        or directly as a method providing the model_name and model class.

        Args:
            model_name (str): name of the model. Must be unique. Same name will be used in
                the config file to refer to the model.
            model_cls (Optional[Type[torch.nn.Module]], optional): model class. Defaults to None.
        """
        if model_name in cls.pretrained_model_name_map:
            raise ValueError(
                f"Pretrained model {model_name} already exists in the registry. "
                f"Please choose a different name."
            )
        if model_cls is None:  # expected when using decorator
            return lambda model_cls: cls.register(model_name, model_cls)

        cls.pretrained_model_name_map[model_name] = model_cls
        return model_cls
