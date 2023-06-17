from typing import Callable, Dict, Optional


class MetricRegistry:
    """Central repository for all metrics.

    - To register a new metric you can use the `add_metric` decorator as follows:
    @MetricRegistry.add_metric('my_new_metric')
    def my_new_metric(inputs, outputs):
    ...

    - Or you can use the `add_metric` method directly:

    def my_new_metric(inputs, outputs):
    ...
    MetricRegistry.add_metric('my_new_metric', my_new_metric)
    """

    metric_name_map: Dict[str, Callable] = {}

    @classmethod
    def register(
        cls,
        metric_name: str,
        metric_fn: Optional[Callable] = None,
    ) -> Callable:
        """Register a new metric. This can be used as a decorator providing only the metric_name
        or directly as a method providing the metric_name and metric function.

        Args:
            metric_name (str): name of the metric. Must be unique. Same name will be used in
                the config file to refer to the metric.
            metric_fn (Optional[Callable], optional): metric function (or any callable).
                Defaults to None.
        """
        if metric_name in cls.metric_name_map:
            raise ValueError(
                f"Metric {metric_name} already exists in the registry. "
                f"Please choose a different name."
            )
        if metric_fn is None:  # expected when using decorator
            return lambda metric_fn: cls.register(metric_name, metric_fn)
        if not callable(metric_fn):
            raise TypeError(
                f"Expected metric {metric_name} to be callable "
                f"but got {type(metric_fn)} instead."
            )
        cls.metric_name_map[metric_name] = metric_fn
        return metric_fn
