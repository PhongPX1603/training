import torch
import argparse
import utils
import time

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='config_yaml/config.yaml')
    parser.add_argument('--weight-path', type=str)
    parser.add_argument('--log-path', type=str)
    args = parser.parse_args()
    
    config = utils.load_yaml(args.config_path)
    stages = utils.eval_config(config=config)
    
    logger = stages['logger']
    logger.log_dir = str(args.log_path)
    logger = logger.get_logger('test')
    
    
    
    valid_loader = stages['data']['valid']
    model = stages['model']
    
    checkpoint = torch.load(args.weight_path)
    state_dict = checkpoint['state_dict']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    metrics = stages['metrics']
    total_loss = 0.0
    
    config = utils.load_yaml(args.config_path)
    stages = utils.eval_config(config=config)
    
    metrics.reset
    print(f'{time.asctime()} - START TESTING')
    logger.info(f'{time.asctime()} - START TESTING')
    with torch.no_grad():
        for params in tqdm(valid_loader):
            params = [param.to(device) for param in params if torch.is_tensor(param)]
            params[0] = model(params[0])
            
            iter_metric = metrics.iteration_compute(
                evaluator_name='valid',
                output=(params)
            )
            metrics.update(iter_metric)
            result = metrics.epoch_compute
    print(f'{time.asctime()} - TESTING COMPLETE !!!')
    logger.info(f"{time.asctime()} - VALID ACCURACY: {result['valid_accuracy']}")
    print(f"{time.asctime()} - VALID ACCURACY: {result['valid_accuracy']}")
    logger.info(f'{time.asctime()} - TESTING COMPLETE !!!')