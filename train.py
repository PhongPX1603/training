import os
import copy
import torch
import argparse
import numpy as np

import utils

from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from trainer.trainer import Trainer


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--config-path', type=str, default='config_yaml/config.yaml')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--project-name', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--save-weight-dir', type=str)
    args = parser.parse_args()
    
    config = utils.load_yaml(args.config_path)
    stages = utils.eval_config(config=config)
    
    train_loader = stages['data']['train']
    train_eval_loader = stages['data']['train_eval']
    valid_loader = stages['data']['valid']
    
    model = stages['model']
    optimizer = stages['optimizer']
    loss_fn = stages['loss']
    metrics = stages['metrics']
    lr_scheduler = stages['lr_scheduler']
    early_stopping = stages['early_stopping']
    writer = SummaryWriter(f'runs/{args.project_name}')
    step = 0
    
    time = datetime.now().strftime(r'%y%m%d%H%M')
    save_weight_dir = Path(f'{args.save_weight_dir} / models / {args.project_name} / {time}')
    if not save_weight_dir.exists():
        save_weight_dir.mkdir(parents=True)

    trainer = Trainer(
        model=model,
        criterion=loss_fn,
        optimizer=optimizer,
        metrics=metrics,
        device=args.device,
        writer=writer,
        step=step
    )

    print('Start Training !!!')
    min_loss = np.inf
    for epoch in range(args.num_epochs):
        train_metrics = trainer.train_epoch(evaluator='train', data_loader=train_loader)
        train_eval_metrics = trainer.eval_epoch(evaluator='train_eval', data_loader=train_eval_loader)
        valid_metrics = trainer.eval_epoch(evaluator='valid', data_loader=valid_loader)
        
        print(f'Epoch #{epoch}')
        print(f"\t {train_metrics}")
        print(f'\t {train_eval_metrics}')
        print(f'\t {valid_metrics}')
        
        lr_scheduler.step(valid_metrics[f'valid_loss'])
        early_stopping(valid_metrics)
        if early_stopping.early_stop:
            break

        if valid_metrics[f'valid_loss'] < min_loss:                
            if save_weight_dir.joinpath(f'best_valid_loss_{min_loss}.pth').exists():
                os.remove(str(save_weight_dir.joinpath(f'best_valid_loss_{min_loss}.pth')))

            model_state_dict = copy.deepcopy(model.state_dict())
            optim_state_dict = copy.deepcopy(optimizer.state_dict())

            min_loss = valid_metrics[f'valid_loss']
            checkpoint = {
                'state_dict': model_state_dict,
                'optimizer': optim_state_dict
            }

            save_path = save_weight_dir.joinpath(f'best_valid_loss_{min_loss}.pth')
            torch.save(obj=checkpoint, f=str(save_path))
