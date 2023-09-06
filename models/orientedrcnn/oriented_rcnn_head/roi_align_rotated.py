from mmcv.ops import nms
import torch
import numpy as np
from collections import OrderedDict
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from ..data_formats import RPNOutput
from ..encoder import encode, Encodings
import os

import orientedrcnn._C as _C

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
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return _C.roi_align_rotated_forward(
                input, rois, self.spatial_scale, output_size[0], output_size[1], self.sampling_ratio
            ).to(dtype=orig_dtype)

        return roi_align_rotated(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        ).to(dtype=orig_dtype)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr

class RoIAlignRotatedWrapper(ROIAlignRotated):
    def __init__(self, output_size, spatial_scale, sampling_ratio: int, fpn_strides = [4, 8, 16, 32, 64]):
        self.fpn_strides = fpn_strides
        super().__init__(
            output_size, 
            spatial_scale, 
            sampling_ratio=sampling_ratio
        )
        self.test = 0

    def process_rpn_proposals(self, rpn_proposals: OrderedDict[str, RPNOutput], anchors: OrderedDict):
        self.test += 1
        result = OrderedDict()

        # transform proposals to vertices
        for k in rpn_proposals.keys():
            result[k] = encode(
                rpn_proposals[k].anchor_offsets,
                Encodings.ANCHOR_OFFSET,
                Encodings.VERTICES,
                anchors[k] 
            )

        # parallelogram to rectangular proposals to cv2_format (center_x, center_y, width, height, theta (rad))
        for k, v in result.items():
            b = rpn_proposals[k].objectness_scores.shape[0]
            cv2_format = encode(v, Encodings.VERTICES, Encodings.ORIENTED_CV2_FORMAT)
            # cv2_format has angle in rad
            cv2_format[..., -1] = cv2_format[..., -1] * 180 / np.pi
            hbb = encode(v, Encodings.VERTICES, Encodings.HBB_CORNERS)
            batch_indexed = []
            level_scores = []
            filtered_vertices = []

            for b_idx in range(b):
                # take the top 2000 rpn proposals and apply nms
                topk_k = min(2000, rpn_proposals[k].objectness_scores.shape[1])
                topk_proposals = torch.topk(rpn_proposals[k].objectness_scores[b_idx], k=topk_k)
                topk_idx = topk_proposals.indices
                topk_scores = topk_proposals.values
                nms_result = nms(hbb[b_idx, topk_idx], topk_scores, 0.5)
                keep = nms_result[1]
                topk_boxes = cv2_format[b_idx, topk_idx]
                kept_boxes = topk_boxes[keep]
                topk_vertices = v[b_idx, topk_idx]
                filtered_vertices.append(topk_vertices[keep])
                kept_scores = topk_scores[keep]
                n = kept_boxes.shape[0]
                b_idx_tensor = torch.full((n, 1), b_idx).to(v.device)
                values = torch.concatenate((b_idx_tensor, kept_boxes), dim=-1)
                batch_indexed.append(values)
                level_scores.append(kept_scores)

            result[k] = {}
            result[k]["boxes"] = torch.concatenate(batch_indexed, dim=0)
            result[k]["scores"] = torch.stack(level_scores, dim=0)
            result[k]["vertices"] = torch.stack(filtered_vertices, dim=0)

        return result

    def forward(self, fpn_features: OrderedDict, rpn_proposals: OrderedDict, anchors: OrderedDict):
        # this is doing the roi align rotated + the filtering described in the section 3.3 of the paper
        cv2_format = self.process_rpn_proposals(rpn_proposals, anchors)
        merged_features = []
        merged_boxes = []
        merged_scores = []
        merged_vertices = []

        for s_idx, k in enumerate(cv2_format.keys()):
            num_batches = fpn_features[k].shape[0]
            roi_align = super().forward(fpn_features[k], cv2_format[k]["boxes"])
            # todo
            roi_align = torch.nan_to_num(roi_align, 0.0)
            # todo: find a better way
            batched_roi_align = {b: [] for b in range(num_batches)}
            batched_boxes = {b: [] for b in range(num_batches)}
            for idx, batch_idx in enumerate(cv2_format[k]["boxes"][:, 0]):
                batched_roi_align[int(batch_idx.item())].append(roi_align[idx])
                batched_boxes[int(batch_idx.item())].append(cv2_format[k]["boxes"][idx, 1:])
            merged_features.append(torch.stack([torch.stack(bra, dim=0) for bra in batched_roi_align.values()], dim=0))
            boxes = torch.stack([torch.stack(bb, dim=0) for bb in batched_boxes.values()], dim=0)
            boxes[..., :-1] = boxes[..., :-1] * self.fpn_strides[s_idx]
            merged_boxes.append(boxes)
            merged_scores.append(cv2_format[k]["scores"])
            merged_vertices.append(cv2_format[k]["vertices"] * self.fpn_strides[s_idx])

        merged_features = torch.concatenate(merged_features, dim=1)
        merged_vertices = torch.concatenate(merged_vertices, dim=1)
        merged_boxes = torch.concatenate(merged_boxes, dim=1)
        merged_scores = torch.concatenate(merged_scores, dim=1)

        filtered_features = []
        filtered_boxes = []
        filtered_scores = []
        num_batches = merged_features.shape[0]
        for b in range(num_batches):
            keep = torch.topk(merged_scores[b], k=1000).indices
            filtered_features.append(merged_features[b, keep])
            filtered_boxes.append(merged_boxes[b, keep])
            filtered_scores.append(merged_scores[b, keep])

        return {
            "features": torch.stack(filtered_features, dim=0), 
            "boxes": torch.stack(filtered_boxes, dim=0), 
            "scores": torch.stack(filtered_scores, dim=0)
        }

