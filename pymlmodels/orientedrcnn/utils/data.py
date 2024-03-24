import torch
from typing import Tuple
import numpy as np 
from .encoder import encode, Encodings

def normalize(
        data: torch.Tensor,
        target_mean: torch.Tensor | Tuple | list | None = None,
        target_std: torch.Tensor | Tuple | list | None = None,
        dim=-1
    ):
    mean = torch.mean(data, dim, keepdim=True)
    std = torch.std(data, dim, keepdim=True)

    if target_mean is not None:
        mean = mean + data.new_tensor(target_mean)
    if target_std is not None:
        std = std / data.new_tensor(target_std)

    data = data - mean
    data = data / std
    return data

def sample_randomly_adjusted_vertices(vertices: torch.Tensor, n_samples=1000):
    r_idx = np.random.randint(0, len(vertices), size=(n_samples,))
    samples = vertices[r_idx]
    midpoint_offset = encode(samples, Encodings.VERTICES, Encodings.MIDPOINT_OFFSET)
    adjusted = randomly_adjust_midpoint_offset(midpoint_offset)
    return encode(adjusted, Encodings.MIDPOINT_OFFSET, Encodings.VERTICES)

def randomly_adjust_midpoint_offset(midpoint_offset: torch.Tensor):
    x = midpoint_offset[..., 0]
    y = midpoint_offset[..., 1]
    w = midpoint_offset[..., 2]
    h = midpoint_offset[..., 3]
    a = midpoint_offset[..., 4]
    b = midpoint_offset[..., 5]

    x = x + w * torch.normal(mean=0, std=0.1, size=(w.shape)).to(x.device)
    y = y + h * torch.normal(mean=0, std=0.1, size=(h.shape)).to(x.device)
    w = w * torch.normal(mean=1, std=0.1, size=(w.shape)).to(x.device)
    h = h * torch.normal(mean=1, std=0.1, size=(h.shape)).to(x.device)
    a = a + w * torch.normal(mean=0, std=0.1, size=(w.shape)).to(x.device)
    b = b + h * torch.normal(mean=0, std=0.1, size=(h.shape)).to(x.device)

    return torch.stack((x, y, w, h, a, b), dim=-1)


