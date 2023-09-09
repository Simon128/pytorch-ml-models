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
    #adjusted = randomly_adjust_midpoint_offset(midpoint_offset)
    return encode(midpoint_offset, Encodings.MIDPOINT_OFFSET, Encodings.VERTICES)

def randomly_adjust_midpoint_offset(midpoint_offset: torch.Tensor):
    x = midpoint_offset[..., 0]
    y = midpoint_offset[..., 1]
    w = midpoint_offset[..., 2]
    h = midpoint_offset[..., 3]
    a = midpoint_offset[..., 4]
    b = midpoint_offset[..., 5]

    torch.manual_seed(12)
    x = x + w * torch.normal(mean=0, std=0.1, size=(w.shape)).to(x.device)
    torch.manual_seed(13)
    y = y + h * torch.normal(mean=0, std=0.1, size=(h.shape)).to(x.device)
    torch.manual_seed(14)
    w = w * torch.normal(mean=1, std=0.1, size=(w.shape)).to(x.device)
    torch.manual_seed(15)
    h = h * torch.normal(mean=1, std=0.1, size=(h.shape)).to(x.device)
    torch.manual_seed(16)
    a = a + w * torch.normal(mean=0, std=0.1, size=(w.shape)).to(x.device)
    torch.manual_seed(17)
    b = b + h * torch.normal(mean=0, std=0.1, size=(h.shape)).to(x.device)

    return torch.stack((x, y, w, h, a, b), dim=-1)


