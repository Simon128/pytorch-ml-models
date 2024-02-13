import torch
import numpy as np
from collections import OrderedDict
from torch import nn
from typing import Tuple
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from ..utils import RPNOutput, encode, Encodings, Annotation, sample_randomly_adjusted_vertices
from torchvision.ops import nms

import pymlmodels.orientedrcnn._C as _C

EPS = 1e-8

class _ROIAlignRotated(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        output = _C.roi_align_rotated_forward(
            input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (rois,) = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_align_rotated_backward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
        )
        return grad_input, None, None, None, None, None


roi_align_rotated = _ROIAlignRotated.apply

class ROIAlignRotated(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        """
        Args:
            output_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.

        Note:
            ROIAlignRotated supports continuous coordinate by default:
            Given a continuous coordinate c, its two neighboring pixel indices (in our
            pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
            c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
            from the underlying signal at continuous coordinates 0.5 and 1.5).
        """
        super(ROIAlignRotated, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx6 boxes. First column is the index into N.
                The other 5 columns are (x_ctr, y_ctr, width, height, angle_degrees).
        """
        assert rois.dim() == 2 and rois.size(1) == 6
        orig_dtype = input.dtype
        if orig_dtype == torch.float16:
            input = input.float()
            rois = rois.float()
        output_size = _pair(self.output_size)

        # Scripting for Autograd is currently unsupported.
        # This is a quick fix without having to rewrite code on the C++ side
        if torch.jit.is_scripting() or torch.jit.is_tracing(): # type: ignore
            return _C.roi_align_rotated_forward(
                input, rois, self.spatial_scale, output_size[0], output_size[1], self.sampling_ratio
            ).to(dtype=orig_dtype)

        return roi_align_rotated(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        ).to(dtype=orig_dtype) # type: ignore

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr

class RoIAlignRotatedWrapper(ROIAlignRotated):
    def __init__(
            self, 
            output_size, 
            spatial_scale, 
            sampling_ratio: int, 
            fpn_strides = [4, 8, 16, 32, 64], 
        ):
        self.fpn_strides = fpn_strides
        super().__init__(
            output_size, 
            spatial_scale, 
            sampling_ratio=sampling_ratio
        )

    def clip_vertices(self, vertices: torch.Tensor, size: Tuple[int, int]):
        x = vertices[..., 0]
        y = vertices[..., 1]
        height, width = size
        x = x.clamp(min=0, max=width)
        y = y.clamp(min=0, max=height)
        return torch.stack((x, y), dim=-1)

    def process_rpn_proposals(
            self, 
            rpn_proposals: list[torch.Tensor]
        ):
        result = [None] * len(rpn_proposals)
        for b_idx in range(len(rpn_proposals)):
            roi_format = encode(rpn_proposals[b_idx], Encodings.VERTICES, Encodings.THETA_FORMAT_BL_RB)
            n = roi_format.shape[0]
            b_idx_tensor = torch.full((n, 1), b_idx, device=roi_format.device)
            result[b_idx] = torch.concatenate((b_idx_tensor, roi_format), dim=-1) # type:ignore
        return torch.concatenate(result, dim=0) # type:ignore

    def forward(
            self, 
            fpn_features: OrderedDict, 
            rpn_proposals: list[torch.Tensor], 
            keys: list[torch.Tensor]
        ):
        device = rpn_proposals[0].device
        channels = fpn_features[0].shape[1]
        output = []
        for b in range(len(rpn_proposals)):
            _output = torch.zeros((len(rpn_proposals[b]),channels,*self.output_size), device=device)
            roi_format = encode(rpn_proposals[b], Encodings.VERTICES, Encodings.THETA_FORMAT_BL_RB)
            b_idx_tensor = torch.full((len(rpn_proposals[b]), 1), 0, device=roi_format.device)
            _in = torch.concatenate((b_idx_tensor, roi_format), dim=-1) # type:ignore

            for k in fpn_features.keys():
                mask = keys[b] == k
                roi_align = super().forward(fpn_features[k], _in[mask])
                _output[mask] = roi_align.float()

            output.append(_output)

        return output

        output = []
        for b in range(len(rpn_proposals)):
            _output = []
            for p, k in zip(rpn_proposals[b], keys[b]):
                roi_format = encode(p, Encodings.VERTICES, Encodings.THETA_FORMAT_BL_RB)
                b_idx_tensor = torch.full((1, 1), b, device=roi_format.device)
                _in = torch.concatenate((b_idx_tensor, roi_format.unsqueeze(0)), dim=-1) # type:ignore
                roi_align = super().forward(fpn_features[k.item()], _in)
                _output.append(roi_align)
            output.append(torch.stack(_output, dim=0))
        return output
