import os
import copy
import torch
import argparse
import numpy as np

import utils

from pathlib import Path
from datetime import datetime
from trainer.trainer import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='config_yaml/config.yaml')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--project-name', type=str)
    parser.add_argument('--num-gpus', type=int)
    parser.add_argument('--save-weight-dir', type=str)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default='saved')
    args = parser.parse_args()
    
    # set up stages
    config = utils.load_yaml(args.config_path)
    stages = utils.eval_config(config=config)
    # data loader
    train_loader = stages['data']['train']
    train_eval_loader = stages['data']['train']
    valid_loader = stages['data']['valid']
    #set up logger
    logger = stages['logger'].get_logger
    # set up for trainer
    model = stages['model']
    logger.info(model)
    optimizer = stages['optimizer']
    loss_fn = stages['loss']
    metrics = stages['metrics']
    lr_scheduler = stages['lr_scheduler']
    early_stopping = stages['early_stopping']
    tensorboard = stages['tensorboard']
    
    # prepare for (multi-device) GPU training
    device, device_ids = utils.prepare_device(n_gpu_use=args.num_gpus)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).module
    # Resume
    if args.resume_path is not None:
        checkpoint = torch.load(f=args.resume_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('RESUME !!!')
    # set up save weight dir
    time = datetime.now().strftime(r'%y%m%d%H%M')
    save_weight_dir = Path(f'{args.save_weight_dir}/models/{args.project_name}/{time}')
    if not save_weight_dir.exists():
        save_weight_dir.mkdir(parents=True)

    trainer = Trainer(
        model=model,
        criterion=loss_fn,
        optimizer=optimizer,
        metrics=metrics,
        device=device,
        tensorboard=tensorboard,
    )

    print('Start Training !!!')
    min_loss = np.inf
    for epoch in range(args.num_epochs):
        train_metrics = trainer.train_epoch(evaluator_name='train', data_loader=train_loader)
        train_eval_metrics = trainer.eval_epoch(evaluator_name='train_eval', data_loader=train_eval_loader)
        valid_metrics = trainer.eval_epoch(evaluator_name='valid', data_loader=valid_loader)
        
        print(f'Epoch #{epoch}')
        print(f"\t {train_metrics}")
        print(f'\t {train_eval_metrics}')
        print(f'\t {valid_metrics}')
        logger.info(train_metrics)
        logger.info(train_eval_metrics)
        logger.info(valid_metrics)
        
        lr_scheduler.step(valid_metrics[f'valid_loss'])
        early_stopping(valid_metrics)
        if early_stopping.early_stop:
            logger.info('Model can not improve. Stop Training !!!')
            break
        
        if save_weight_dir.joinpath(f'best_valid_loss_{min_loss}.pth').exists():
                os.remove(str(save_weight_dir.joinpath(f'best_valid_loss_{min_loss}.pth')))
        model_state_dict = copy.deepcopy(model.state_dict())
        optim_state_dict = copy.deepcopy(optimizer.state_dict())
        backup = {
            'state_dict': model_state_dict,
            'optimizer': optim_state_dict
        }
        save_path = save_weight_dir.joinpath(f'backup_epoch{epoch}.pth')
        logger.info(f'Saving Back_up: {str(save_path)}')
        torch.save(obj=backup, f=str(save_path))

        if valid_metrics['valid_loss'] < min_loss:                
            if save_weight_dir.joinpath(f'best_valid_loss_{min_loss}.pth').exists():
                os.remove(str(save_weight_dir.joinpath(f'best_valid_loss_{min_loss}.pth')))
            checkpoint = {
                'state_dict': model_state_dict
            }
                
            min_loss = valid_metrics[f'valid_loss']
            save_path = save_weight_dir.joinpath(f'best_valid_loss_{min_loss}.pth')
            logger.info(f'Saving Checkpoint: {str(save_path)}')
            torch.save(obj=checkpoint, f=str(save_path))
