import os
import copy
import torch
import logging
import argparse
import numpy as np

import utils

from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from trainer.trainer import Trainer


def prepare_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')    # to save checkpoint, train: just save model state dict - resume: save both model and optimizer state dict
    parser.add_argument('--config-path', type=str, default='config_yaml/config.yaml')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--project-name', type=str)
    parser.add_argument('--num-gpus', type=int)
    parser.add_argument('--save-weight-dir', type=str)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default='saved')
    args = parser.parse_args()
    
    #set up logger
    log_dir = args.log_dir
    if not Path(log_dir).exists():
        Path(log_dir).mkdir()
    
    logger = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)
    file_hand = logging.FileHandler(f'{log_dir}/file.log')
    formatter = logging.Formatter(f'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_hand.setFormatter(formatter)
    logger.addHandler(file_hand)
    
    # set up
    config = utils.load_yaml(args.config_path)
    stages = utils.eval_config(config=config)
    # data loader
    train_loader = stages['data']['train']
    train_eval_loader = stages['data']['train']
    valid_loader = stages['data']['valid']
    # set up for trainer
    model = stages['model']
    logger.info(model)
    optimizer = stages['optimizer']
    loss_fn = stages['loss']
    metrics = stages['metrics']
    lr_scheduler = stages['lr_scheduler']
    early_stopping = stages['early_stopping']
    writer = SummaryWriter(f'runs/{args.project_name}')
    step = 0
    # Resume
    if args.resume_path is not None:
        checkpoint = torch.load(f=args.resume_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('RESUME !!!')
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(args.num_gpus)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    # set up save weight dir
    time = datetime.now().strftime(r'%y%m%d%H%M')
    save_weight_dir = Path(f'{args.save_weight_dir}/models/{args.project_name}/{args.mode}/{time}')
    if not save_weight_dir.exists():
        save_weight_dir.mkdir(parents=True)

    trainer = Trainer(
        model=model,
        criterion=loss_fn,
        optimizer=optimizer,
        metrics=metrics,
        device=device,
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
        logger.info(train_metrics)
        logger.info(train_eval_metrics)
        logger.info(valid_metrics)
        
        lr_scheduler.step(valid_metrics[f'valid_loss'])
        early_stopping(valid_metrics)
        if early_stopping.early_stop:
            logger.info('Model can not improve. Stop Training !!!')
            break

        if valid_metrics['valid_loss'] < min_loss:                
            if save_weight_dir.joinpath(f'best_valid_loss_{min_loss}.pth').exists():
                os.remove(str(save_weight_dir.joinpath(f'best_valid_loss_{min_loss}.pth')))

            model_state_dict = copy.deepcopy(model.state_dict())
            # set Resume mode
            if args.mode == 'resume':
                optim_state_dict = copy.deepcopy(optimizer.state_dict())
                checkpoint = {
                    'state_dict': model_state_dict,
                    'optimizer': optim_state_dict
                }
            elif args.mode == 'train':
                checkpoint = {
                    'state_dict': model_state_dict
                }
                
            min_loss = valid_metrics[f'valid_loss']
            save_path = save_weight_dir.joinpath(f'best_valid_loss_{min_loss}.pth')
            logger.info(f'Saving Checkpoint: {str(save_path)}')
            torch.save(obj=checkpoint, f=str(save_path))
