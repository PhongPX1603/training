import torch
import torch.nn as nn

from collections import defaultdict
from typing import Callable, Dict


class Trainer:
    def __init__(
        self,
        model: nn.Module = None,
        criterion = None,
        optimizer = None,
        metrics = None,
        device: str = 'cpu',
        tensorboard: Callable = None,
    ):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.metrics = metrics
        self.tensorboard = tensorboard
        self.iter_counters = defaultdict(int)

    def train_epoch(self, evaluator_name: str = 'train', data_loader: nn.Module = None) -> Dict[str, float]:
        self.model.to(self.device).train()
        self.metrics.reset
        for samples, targets in data_loader:
            samples = samples.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(samples)
            loss = self.criterion(preds, targets)
            loss.backward()
            self.optimizer.step()

            iter_metric = self.metrics.iteration_compute(
                evaluator_name=evaluator_name,
                output=(preds, targets))
            self.metrics.update(metric=iter_metric)    

            for metric_name, metric_value in iter_metric.items():
                self.tensorboard.write(
                    metric_name,
                    metric_value,
                    step=self.iter_counters[evaluator_name]
                )

            self.iter_counters[evaluator_name] += 1

        return self.metrics.epoch_compute
    
    def eval_epoch(self, evaluator_name: str, data_loader: nn.Module = None) -> Dict[str, float]:
        self.model.to(self.device).eval()
        self.metrics.reset
        with torch.no_grad():
            for samples, targers in data_loader:
                samples = samples.to(self.device)
                targets = targers.to(self.device)
                
                preds = self.model(samples)
                
                iter_metric = self.metrics.iteration_compute(
                    evaluator_name=evaluator_name,
                    output=(preds, targets)
                )
                self.metrics.update(iter_metric)
                for metric_name, metric_value in iter_metric.items():
                    self.tensorboard.write(
                        metric_name,
                        metric_value,
                        step=self.iter_counters[evaluator_name]
                    )

                self.iter_counters[evaluator_name] += 1

        return self.metrics.epoch_compute
