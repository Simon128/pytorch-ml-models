import math
import torch
import torch.nn as nn
from dataclasses import dataclass

class BalancedSampler(nn.Module):
    def __init__(
            self, 
            n_samples: int, 
            pos_fraction: float, 
            neg_thr: float,
            pos_thr: float,
            sample_max_inbetween_as_pos: bool
        ):
        super().__init__()
        self.n_samples = n_samples
        self.n_positives = math.ceil(n_samples * pos_fraction)
        self.neg_thr = neg_thr
        self.pos_thr = pos_thr
        self.sample_max_inbetween_as_pos = sample_max_inbetween_as_pos
        self.mem = []

    def forward(self, iou: torch.Tensor):
        pos_indices = self.__pos_indices(iou)
        neg_indices = self.__neg_indices(iou)

        available_pos = len(pos_indices[0])
        available_neg = len(neg_indices)
        num_pos = min(self.n_positives, available_pos)
        num_neg = min(self.n_samples - num_pos, available_neg)

        choice_pos = torch.randperm(available_pos, device=iou.device)[:num_pos]
        choice_neg = torch.randperm(available_neg, device=iou.device)[:num_neg]
        pos_indices = (pos_indices[0][choice_pos], pos_indices[1][choice_pos])
        neg_indices = neg_indices[choice_neg]

        return pos_indices, neg_indices

    def __pos_indices(self, iou: torch.Tensor):
        max_tensors, max_per_gt = iou.max(dim=1)
        mask = max_tensors > self.pos_thr

        if self.sample_max_inbetween_as_pos:
            inbetween = (max_tensors > self.neg_thr) & (max_tensors < self.pos_thr)
            mask = mask | inbetween

        return (mask.nonzero(as_tuple=True)[0], max_per_gt[mask])

    def __neg_indices(self, iou: torch.Tensor):
        return ((iou < self.neg_thr).sum(dim=1) == iou.shape[1]).nonzero(as_tuple=True)[0]
