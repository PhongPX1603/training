from statistics import mode
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        mode='min',
        monitor='valid_loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            mode (str): 'min' - use with valid loss and 'max' use for valid acc
            monitor (str): 'valid_loss' or 'valid_accuracy' to take value in metrics
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.mode = mode
        self.monitor = monitor
        
        
    def __call__(self, metrics):
        score = metrics[self.monitor]
        if self.mode == 'max':
            score = -score

        if self.best_score is None:
            self.best_score = score
            
        elif score > self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print('Training Stop !!!')
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0