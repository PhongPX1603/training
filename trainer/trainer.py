import os
import torch
import copy
import time
import torch.nn as nn
import numpy as np

from collections import defaultdict
from typing import Callable, Dict
from pathlib import Path


class Trainer:
    def __init__(
        self,
        model: nn.Module = None,
        criterion = None,
        optimizer = None,
        metrics = None,
        device: str = 'cpu',
        tensorboard: Callable = None,
        logger = None,
        resume_path: str = None,
        save_dir: Path = None,
        num_epochs: int = None,
        train_loader = None,
        train_eval_loader = None,
        valid_loader = None,
        lr_scheduler = None,
        early_stopping = None,
    ):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.metrics = metrics
        self.tensorboard = tensorboard
        self.iter_counters = defaultdict(int)
        self.logger = logger
        self.resume_path = resume_path
        self.save_dir = save_dir
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.train_eval_loader = train_eval_loader
        self.valid_loader = valid_loader
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        

    def train_epoch(self, evaluator_name: str = 'train', data_loader: nn.Module = None) -> Dict[str, float]:
        self.model.to(self.device).train()
        self.metrics.reset
        for params in data_loader:
            self.optimizer.zero_grad()
            params = [param.to(self.device) for param in params if torch.is_tensor(param)]
            params[0] = self.model(params[0])
            loss = self.criterion(*params)
            loss.backward()
            self.optimizer.step()

            iter_metric = self.metrics.iteration_compute(
                evaluator_name=evaluator_name,
                output=(params))
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
            for params in data_loader:
                params = [param.to(self.device) for param in params if torch.is_tensor(param)]
                params[0] = self.model(params[0])
                
                iter_metric = self.metrics.iteration_compute(
                    evaluator_name=evaluator_name,
                    output=(params)
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

    def train(self):
        # Resume
        mode = self.early_stopping.mode
        best_score_mode = 1 if mode == 'min' else -1
        if self.resume_path is not None:
            checkpoint = torch.load(f=self.resume_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            best_score = checkpoint['best_score']
            score_name = checkpoint['score_name']
            start_epoch = checkpoint['start_epoch']
            self.early_stopping.best_score = best_score
            self.logger.info(f"Start Training with lr - {self.optimizer['lr']}")
            print('RESUME !!!')
        else:
            start_epoch = 0
            score_name = self.early_stopping.monitor
            best_score = np.Inf if mode == 'min' else 0
            print('Start Training !!!')
        
            
        print(f'{time.asctime()} - STARTED')
        for epoch in range(start_epoch, self.num_epochs):
            train_metrics = self.train_epoch(evaluator_name='train', data_loader=self.train_loader)
            train_eval_metrics = self.eval_epoch(evaluator_name='train_eval', data_loader=self.train_eval_loader)
            valid_metrics = self.eval_epoch(evaluator_name='valid', data_loader=self.valid_loader)
            
            print(f'Epoch #{epoch} - {time.asctime()}')
            print(f"\t {train_metrics}")
            print(f'\t {train_eval_metrics}')
            print(f'\t {valid_metrics}')
            self.logger.info(train_metrics)
            self.logger.info(train_eval_metrics)
            self.logger.info(valid_metrics)
            
            self.lr_scheduler.step(valid_metrics[score_name])
            self.early_stopping(valid_metrics)
            if self.early_stopping.early_stop:
                self.logger.info('Model can not improve. Stop Training !!!')
                break
            
            model_state_dict = copy.deepcopy(self.model.state_dict())
            optim_state_dict = copy.deepcopy(self.optimizer.state_dict())
            
            #best checkpoint
            if valid_metrics[score_name] * best_score_mode < best_score:                
                if self.save_dir.joinpath(f'best_{score_name}_{best_score}.pth').exists():
                    os.remove(str(self.save_dir.joinpath(f'best_{score_name}_{best_score}.pth')))
                best_score = valid_metrics[score_name]
                save_path = self.save_dir.joinpath(f'best_{score_name}_{best_score}.pth')
                self.logger.info(f'Saving Checkpoint: {str(save_path)}')
                torch.save(obj=model_state_dict, f=str(save_path))
            
            # back_up checkpoint
            if self.save_dir.joinpath(f'backup_epoch{epoch-1}.pth').exists():
                os.remove(str(self.save_dir.joinpath(f'backup_epoch{epoch-1}.pth')))
            save_path = self.save_dir.joinpath(f'backup_epoch{epoch}.pth')
            self.logger.info(f'Saving Back_up: {str(save_path)}')
            backup = {
                'start_epoch': epoch + 1,
                'state_dict': model_state_dict,
                'optimizer': optim_state_dict,
                'best_score': best_score,
                'score_name': self.early_stopping.monitor,
            }
            torch.save(obj=backup, f=str(save_path))
        print(f'{time.asctime()} - COMPLETE !!!')
        self.logger.info(f'{time.asctime()} - COMPLETE !!!')
