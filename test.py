import torch
import argparse
import utils

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='config_yaml/config.yaml')
    parser.add_argument('--weight-path', type=str)
    args = parser.parse_args()
    
    config = utils.load_yaml(args.config_path)
    stage = utils.eval_config(config=config)
    
    valid_loader = stage['data']['valid_loader']
    model = stage['model']
    
    checkpoint = torch.load(args.weight_path)
    state_dict = checkpoint['state_dict']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    metrics = stage['metrics']
    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))
    
    criterion = stage['loss']
    
    config = utils.load_yaml(args.config_path)
    stage = utils.eval_config(config=config)
    
    accuracies, losses = [], []
    with torch.no_grad():
        for i, (samples, targets) in enumerate(tqdm(valid_loader)):
            samples, targets = samples.to(device), targets.to(device)
            preds = model(samples)
            correct = torch.sum(targets == preds.argmax(dim=1))
            accuracy = correct.item() / targets.shape[0]
            
            accuracies.append(accuracy)
        
    accuracy = sum(accuracies) / len(accuracies) if len(accuracies) else 0
    print(f'accuracy: {accuracy * 100: .2f}%')