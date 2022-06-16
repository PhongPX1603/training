import torch
import argparse

import utils

from pathlib import Path
from trainer.trainer import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='config_yaml/config.yaml')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-gpus', type=int)
    # parser.add_argument('--save-dir', type=str)
    parser.add_argument('--resume-path', type=str, default=None)
    args = parser.parse_args()
    
    # set up stages
    config = utils.load_yaml(args.config_path)
    stages = utils.eval_config(config=config)
    #set up tensorboard
    tensorboard = stages['tensorboard']
    # set up save training files dir
    # save_dir = Path(f'{args.save_dir}')
    # save_dir = save_dir.joinpath(max(os.listdir(save_dir)))
    # if not save_dir.exists():
    #     save_dir.mkdir(parents=True)
    save_dir = Path(tensorboard.tb_dir)
    
    # data loader
    train_loader = stages['data']['train']
    train_eval_loader = stages['data']['train_eval']
    valid_loader = stages['data']['valid']
    #set up logger
    logger = stages['logger']
    logger.log_dir = str(save_dir)
    logger = logger.get_logger('train')
    # set up for trainer
    model = stages['model']
    logger.info(model)
    optimizer = stages['optimizer']
    loss_fn = stages['loss']
    metrics = stages['metrics']
    lr_scheduler = stages['lr_scheduler']
    early_stopping = stages['early_stopping']
    
    # prepare for (multi-device) GPU training
    device, device_ids = utils.prepare_device(n_gpu_use=args.num_gpus)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).module

    trainer = Trainer(
        model=model,
        criterion=loss_fn,
        optimizer=optimizer,
        metrics=metrics,
        device=device,
        tensorboard=tensorboard,
    )
    
    trainer.train(
        resume_path=args.resume_path,
        save_dir=save_dir,
        num_epochs=args.num_epochs,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        valid_loader=valid_loader,
        lr_scheduler=lr_scheduler,
        early_stopping=early_stopping,
        logger=logger
        )
