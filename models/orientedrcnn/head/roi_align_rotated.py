import torch
import numpy as np
from collections import OrderedDict
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from ..utils import RPNOutput, encode, Encodings, Annotation, sample_randomly_adjusted_vertices
from torchvision.ops import nms

import orientedrcnn._C as _C

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
            inject_annotation = False,
            n_injected_samples: int = 1000
        ):
        self.fpn_strides = fpn_strides
        self.inject_annotation = inject_annotation
        self.n_injected_samples = n_injected_samples
        super().__init__(
            output_size, 
            spatial_scale, 
            sampling_ratio=sampling_ratio
        )

    def process_rpn_proposals(
            self, 
            rpn_proposals: OrderedDict[str, RPNOutput], 
            anchors: OrderedDict, 
            annotation: Annotation | None = None,
            reduce_injected_samples: int = 0
        ):
        result = OrderedDict()
        # transform proposals to vertices
        for i, k in enumerate(rpn_proposals.keys()):
            result[k] = dict()
            # detach rpn proposals to prevent
            # gradient of head to flow back to RPN specific layers
            lve = encode(
                rpn_proposals[k].anchor_offsets.detach().clone(),
                Encodings.ANCHOR_OFFSET,
                Encodings.VERTICES,
                anchors[k],
            )
            if self.inject_annotation and self.n_injected_samples - reduce_injected_samples > 0:
                assert annotation is not None, "missing ground truth"
                n_samples = self.n_injected_samples - reduce_injected_samples
                shape = list(lve.shape)
                shape[1] = n_samples
                result[k]["vertices"] = torch.cat((lve, torch.zeros(shape).to(lve.device)), dim=1)
                shape = list(rpn_proposals[k].objectness_scores.shape)
                shape[1] = n_samples
                result[k]["objectness"] = torch.cat(
                    (
                        rpn_proposals[k].objectness_scores.detach().clone(), 
                        # 100 for sigmoid -> close to 1
                        torch.full(shape, 10000, dtype=torch.float).to(rpn_proposals[k].objectness_scores.device)
                    ),
                    dim=1
                )

                for b in range(len(annotation.boxes)):
                    boxes = sample_randomly_adjusted_vertices(annotation.boxes[b] / self.fpn_strides[i], n_samples)
                    result[k]["vertices"][b, -n_samples:] = boxes
            else:
                result[k]["vertices"] = lve
                result[k]["objectness"] = rpn_proposals[k].objectness_scores.detach().clone

        for test_i , (k, v) in enumerate(result.items()):
            b = result[k]["objectness"].shape[0]
            cv2_format = encode(v["vertices"], Encodings.VERTICES, Encodings.ORIENTED_CV2_FORMAT)
            # cv2_format has angle in rad
            cv2_format[..., -1] = -1* cv2_format[..., -1] * 180 / np.pi
            #debug
            #import cv2
            #test_box = cv2_format[0, -1]
            #test_box[..., :-1] = test_box[..., :-1] * self.fpn_strides[test_i]
            #test_box2 = torch.cat((torch.zeros((1,)).to(test_box.device), test_box))
            #test_box2 = test_box2.unsqueeze(0)
            #image = cv2.imread("/home/simon/unibw/pytorch-ml-models/models/orientedrcnn/example/image.tif")
            #tensor = torch.tensor(image).permute((2, 0, 1)).unsqueeze(0).to(test_box.device)
            #_roi = ROIAlignRotated(
            #    output_size=(test_box[3], test_box[2]), 
            #    spatial_scale=1, 
            #    sampling_ratio=1
            #)
            #z = _roi.forward(tensor.float(), test_box2)
            #z_np = z.detach().cpu().squeeze(0).permute((1, 2, 0)).numpy()
            #cv2.imwrite("test.png", z_np) # type:ignore
            #rot = ((test_box2[0, 1].item(), test_box2[0,2].item()), (test_box2[0,3].item(), test_box2[0,4].item()), -1*test_box2[0,5].item())
            #pts = cv2.boxPoints(rot) # type: ignore
            #pts = np.intp(pts) 
            #image_check = cv2.drawContours(image.copy(), [pts], 0, (0, 255, 0), 2) # type: ignore
            #cv2.imwrite("test_check.png", image_check) # type:ignore
            #
            hbb = encode(v["vertices"], Encodings.VERTICES, Encodings.HBB_CORNERS)
            batch_indexed = []
            level_scores = []

            for b_idx in range(b):
                # take the top 2000 rpn proposals and apply nms
                topk_k = min(2000, result[k]["objectness"].shape[1])
                topk_proposals = torch.topk(result[k]["objectness"][b_idx], k=topk_k)
                topk_idx = topk_proposals.indices
                topk_scores = topk_proposals.values
                keep = nms(hbb[b_idx, topk_idx], topk_scores, 0.5)
                topk_boxes = cv2_format[b_idx, topk_idx]
                kept_boxes = topk_boxes[keep]
                kept_scores = topk_scores[keep]
                n = kept_boxes.shape[0]
                b_idx_tensor = torch.full((n, 1), b_idx).to(v["vertices"].device)
                values = torch.concatenate((b_idx_tensor, kept_boxes), dim=-1)
                batch_indexed.append(values)
                level_scores.append(kept_scores)

            result[k] = {}
            result[k]["boxes"] = torch.concatenate(batch_indexed, dim=0)
            result[k]["scores"] = level_scores

        return result

    def forward(
            self, 
            fpn_features: OrderedDict, 
            rpn_proposals: OrderedDict, 
            anchors: OrderedDict, 
            annotation: Annotation | None = None,
            reduce_injected_samples: int = 0
        ):
        # this is doing the roi align rotated + the filtering described in the section 3.3 of the paper
        num_batches = list(fpn_features.values())[0].shape[0]
        cv2_format = self.process_rpn_proposals(rpn_proposals, anchors, annotation, reduce_injected_samples)
        merged_features = {b: [] for b in range(num_batches)}
        merged_boxes = {b: [] for b in range(num_batches)}
        merged_scores = {b: [] for b in range(num_batches)}
        merged_strides = {b: [] for b in range(num_batches)}

        for s_idx, k in enumerate(cv2_format.keys()):
            roi_align = super().forward(fpn_features[k], cv2_format[k]["boxes"])
            # todo
            roi_align = torch.nan_to_num(roi_align, EPS, EPS, EPS)
            # todo: find a better way

            batched_boxes = {b: [] for b in range(num_batches)}

            for idx, batch_idx in enumerate(cv2_format[k]["boxes"][:, 0]):
                merged_features[int(batch_idx.item())].append(roi_align[idx])
                batched_boxes[int(batch_idx.item())].append(cv2_format[k]["boxes"][idx, 1:])

            for batch_idx in batched_boxes.keys():
                boxes = torch.stack(batched_boxes[batch_idx], dim=0)
                #boxes[..., :-1] = boxes[..., :-1] * self.fpn_strides[s_idx]
                merged_strides[batch_idx].append(torch.full_like(boxes, self.fpn_strides[s_idx]))
                merged_boxes[batch_idx].append(boxes)
                merged_scores[batch_idx].append(cv2_format[k]["scores"][batch_idx])

        for b in range(num_batches):
            merged_features[b] = torch.stack(merged_features[b], dim=0)
            merged_boxes[b] = torch.cat(merged_boxes[b], dim=0)
            merged_strides[b] = torch.cat(merged_strides[b], dim=0)
            merged_scores[b] = torch.cat(merged_scores[b], dim=0)

        filtered_features = []
        filtered_boxes = []
        filtered_scores = []
        filtered_strides = []
        for b in range(num_batches):
            topk_k = min(1000, len(merged_scores[b]))
            keep = torch.topk(merged_scores[b], k=topk_k).indices
            filtered_features.append(merged_features[b][keep])
            filtered_boxes.append(merged_boxes[b][keep])
            filtered_scores.append(merged_scores[b][keep])
            filtered_strides.append(merged_strides[b][keep])

        return {
            "features": torch.stack(filtered_features, dim=0), 
            "boxes": torch.stack(filtered_boxes, dim=0), 
            "scores": torch.stack(filtered_scores, dim=0),
            "strides": torch.stack(filtered_strides, dim=0)
        }
