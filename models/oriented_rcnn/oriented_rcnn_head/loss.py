import torch
import torch.nn.functional as F

def classification_loss(prediction: torch.Tensor, ground_truth: torch.Tensor):
    return F.cross_entropy(prediction, ground_truth)


def regression_loss(prediction: torch.Tensor, ground_truth: torch.Tensor):
    pass
