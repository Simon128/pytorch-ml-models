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

    def forward(self, iou: torch.Tensor, not_training: bool = False):
        pos_indices = self.__pos_indices(iou)
        neg_indices = self.__neg_indices(iou)

        available_pos = len(pos_indices[0])
        available_neg = len(neg_indices[0])
        num_pos = min(self.n_positives, available_pos)
        num_neg = min(self.n_samples - num_pos, available_neg)

        if not not_training:
            choice_pos = torch.randperm(available_pos, device=iou.device)[:num_pos]
            choice_neg = torch.randperm(available_neg, device=iou.device)[:num_neg]
            pos_indices = (pos_indices[0][choice_pos], pos_indices[1][choice_pos])
            neg_indices = (neg_indices[0][choice_neg], neg_indices[1][choice_neg])

        return pos_indices, neg_indices

    def __pos_indices(self, iou: torch.Tensor):
        above = (iou > self.pos_thr).nonzero(as_tuple=True)

        if not self.sample_max_inbetween_as_pos:
            return above

        max_per_gt = torch.max(iou, dim=0, keepdim=True).indices
        max_tensors = torch.gather(iou, 0, max_per_gt)
        inbetween = (max_tensors > self.neg_thr) & (max_tensors < self.pos_thr)
        inbetween = inbetween.nonzero(as_tuple=True)
        inbetween = (max_per_gt[inbetween], inbetween[1])

        if len(above[0]) == 0:
            return inbetween
        elif len(inbetween[0]) == 0:
            return above

        return (torch.cat((above[0], inbetween[0])), torch.cat((above[1], inbetween[1])))

    def __neg_indices(self, iou: torch.Tensor):
        return (iou < self.neg_thr).nonzero(as_tuple=True)
