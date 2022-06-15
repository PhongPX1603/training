import torch
from collections import defaultdict
from typing import Callable, Dict, Any, Tuple


class Metric:
    def __init__(self, metric: Dict[str, Callable], output_transform: Callable = lambda x: x):
        super(Metric, self).__init__()
        self.metric = metric
        self.output_transform = output_transform
        self.metric_tracker = defaultdict(list)

    @property
    def reset(self):
        self.metric_tracker = defaultdict(list)

    def update(self, metric: Dict[str, float]) -> None:
        for metric_name, metric_value in metric.items():
            self.metric_tracker[metric_name].append(metric_value)

    def iteration_compute(self, evaluator_name: str, output: Tuple[Any]) -> Dict[str, float]:
        iter_metric = dict()
        output = self.output_transform(output)
        for metric_name, metric_fn in self.metric.items():
            value = metric_fn(*output)
            if isinstance(value, torch.Tensor):
                value = value.item()
            iter_metric[f'{evaluator_name}_{metric_name}'] = value

        return iter_metric

    @property
    def epoch_compute(self) -> Dict[str, float]:
        result = {}
        for metric_name, metric_values in self.metric_tracker.items():
            result[metric_name] = sum(metric_values) / len(metric_values)

        return result
