import torch
import torch.nn as nn


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        with torch.no_grad():
            pred = torch.argmax(pred, dim=1)
            assert pred.shape[0] == len(target)
            correct = 0
            correct += torch.sum(pred == target).item()
        return correct / len(target)

class TopKAccuracy(nn.Module):
    def __init__(self):
        super(TopKAccuracy, self).__init__()
    
    def forward(self, pred, target, k=3):
        with torch.no_grad():
            pred = torch.topk(pred, k, dim=1)[1]
            assert pred.shape[0] == len(target)
            correct = 0
            for i in range(k):
                correct += torch.sum(pred[:, i] == target).item()
        return correct / len(target)
