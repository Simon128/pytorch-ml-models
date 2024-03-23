from dataclasses import dataclass
import torch

@dataclass
class BBAVectorAnnotation:
    target_orientation: torch.Tensor
    target_heatmap: torch.Tensor
    target_offset: torch.Tensor
    target_mask: torch.Tensor
    target_index: torch.Tensor
    target_vector: torch.Tensor

@dataclass
class BBAVectorInput:
    image: torch.Tensor
    annotation: BBAVectorAnnotation | None = None

@dataclass
class BBAVectorLoss:
    total_loss: torch.Tensor
    heatmap_loss: torch.Tensor
    offset_loss: torch.Tensor
    vector_loss: torch.Tensor
    orientation_loss: torch.Tensor

@dataclass
class BBAVectorOutput:
    heatmap: torch.Tensor
    orientation_map: torch.Tensor
    vector_map: torch.Tensor
    offset_map: torch.Tensor
    loss: BBAVectorLoss | None = None
