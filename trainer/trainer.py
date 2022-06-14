import torch
import torch.nn as nn

from datetime import datetime
from pathlib import Path
from typing import Dict

class Trainer:
    def __init__(
        self,
        model: nn.Module = None,
        criterion = None,
        optimizer = None,
        metrics = None,
        device: str = 'cpu',
        writer = None,
        step: int = 0
    ):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.metrics = metrics
        self.writer = writer
        self.step = step
        
    def train_epoch(self, evaluator: str = 'train', data_loader: nn.Module = None) -> Dict[str, float]:
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

            self.metrics.update(
                evaluator=evaluator,
                output=(preds, targets)
            )
            self.writer.add_scalar('training Loss', loss, global_step=self.step)
            acc = self.metrics.metric_tracker[f'{evaluator}_accuracy'][-1]
            self.writer.add_scalar('training Acc', acc, global_step=self.step)
            self.step += 1

        return self.metrics.compute
    
    def eval_epoch(self, evaluator: str, data_loader) -> Dict[str, float]:
        self.model.to(self.device).eval()
        self.metrics.reset
        with torch.no_grad():
            for samples, targers in data_loader:
                samples = samples.to(self.device)
                targets = targers.to(self.device)
                
                preds = self.model(samples)
                
                self.metrics.update(
                    evaluator=evaluator,
                    output=(preds, targets)
                )

        return self.metrics.compute
