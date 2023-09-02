from mmcv.ops import RoIAlignRotated, nms
import torch
from collections import OrderedDict

from ..encodings import anchor_offset_to_midpoint_offset, midpoint_offset_to_vertices

class RoIAlignRotatedWrapper(RoIAlignRotated):
    def __init__(self, output_size, spatial_scale, sampling_ratio: int, fpn_level_scalings = [4, 8, 16, 32, 64]):
        self.fpn_level_scalings = fpn_level_scalings
        super().__init__(output_size, spatial_scale, sampling_ratio=sampling_ratio, clockwise=True)

    def parallelogram_vertices_to_rectangular_vertices(self, parallelogram: torch.Tensor):
        # we get the vectors of both diagonales,
        # normalize them by length
        # and for the shorter diagonal we add the corresponding norm. vector
        # to both endpoints (vertices) scaled by the diag. length difference / 2
        b, n, _, h, w = parallelogram.shape
        num_rois = int(n/4)
        r_parall = parallelogram.reshape((b, num_rois, 4, 2, h, w))

        v1 = r_parall[:, :, 0, :, :, :]
        v2 = r_parall[:, :, 1, :, :, :]
        v3 = r_parall[:, :, 2, :, :, :]
        v4 = r_parall[:, :, 3, :, :, :]

        diag1_len = torch.sqrt(
            torch.square(v3[:, :, 0, :, :] - v1[:, :, 0, :, :]) + 
            torch.square(v3[:, :, 1, :, :] - v1[:, :, 1, :, :])
        ).unsqueeze(2).repeat((1, 1, 2, 1, 1))
        diag2_len = torch.sqrt(
            torch.square(v4[:, :, 0, :, :] - v2[:, :, 0, :, :]) + 
            torch.square(v4[:, :, 1, :, :] - v2[:, :, 1, :, :])
        ).unsqueeze(2).repeat((1, 1, 2, 1, 1))
    
        # assume diag1_len > diag2_len
        # extend diag2
        extension_len = (diag1_len - diag2_len) / 2
        norm_ext_vector = (v4 - v2) / diag2_len
        new_v4 = v4 + norm_ext_vector * extension_len
        new_v2 = v2 + -1 * norm_ext_vector * extension_len

        # assume diag1_len < diag2_len
        # extend diag1
        extension_len = (diag2_len - diag1_len) / 2
        norm_ext_vector = (v3 - v1) / diag1_len
        new_v3 = v3 + norm_ext_vector * extension_len
        new_v1 = v1 + -1 * norm_ext_vector * extension_len

        v1_new = torch.where(diag1_len > diag2_len, v1, new_v1)
        v2_new = torch.where(diag1_len > diag2_len, new_v2, v2)
        v3_new = torch.where(diag1_len > diag2_len, v3, new_v3)
        v4_new = torch.where(diag1_len > diag2_len, new_v4, v4)

        r_rectangular = torch.stack((v1_new, v2_new, v3_new, v4_new), dim=2)
        return torch.cat([r_rectangular[:, i, :, :, :, :] for i in range(num_rois)], dim=1)

    def rectangular_vertices_to_5_param_and_hbb(self, rect_v: torch.Tensor):
        # transform rectangular vertices to (x, y, w, h, theta)
        # with x,y being center coordinates of box and theta 
        # correponding to the theta as defined by the mmcv RoiAlignRotated 
        b, n, _, h, w = rect_v.shape
        num_rois = int(n/4)
        r_rect = rect_v.reshape((b, num_rois, 4, 2, h, w))
        r_rect = torch.movedim(r_rect, (2, 3), (4, 5))
        # clockwise assumption
        # (first min_y will be the left one if there are two)
        min_y_idx = torch.argmin(r_rect[:, :, :, :, :, 1], dim=4, keepdim=True)
        min_y_tensors = torch.gather(r_rect, 4, min_y_idx.unsqueeze(-1).repeat((1, 1, 1, 1, 1, 2)))
        # for the reference vector, we need the correct neighbouring vertex 
        # which is the one with largest x coord
        max_x_idx = torch.argmax(r_rect[:, :, :, :, :, 0], dim=4, keepdim=True)
        max_x_tensors = torch.gather(r_rect, 4, max_x_idx.unsqueeze(-1).repeat((1, 1, 1, 1, 1, 2)))
        ref_vector = max_x_tensors - min_y_tensors
        angle = torch.arccos(ref_vector[:, :, :, :, :, 0] / (torch.norm(ref_vector, dim=-1) + 1))
        width = max_x_tensors[:, :, :, :, :, 0] - min_y_tensors[:, :, :, :, :, 0]
        x_center = min_y_tensors[:, :, :, :, :, 0] + width/2
        max_y_idx = torch.argmax(r_rect[:, :, :, :, :, 1], dim=4, keepdim=True)
        max_y_tensors = torch.gather(r_rect, 4, max_y_idx.unsqueeze(-1).repeat((1, 1, 1, 1, 1, 2)))
        height =  max_y_tensors[:, :, :, :, :, 1] - min_y_tensors[:, :, :, :, :, 1]
        y_center = min_y_tensors[:, :, :, :, :, 1] + height / 2
        five_params = torch.stack((x_center, y_center, width, height, angle), dim=-1).reshape((b, -1, 5))
        min_x_idx = torch.argmax(r_rect[:, :, :, :, :, 0], dim=4, keepdim=True)
        min_x_tensors = torch.gather(r_rect, 4, min_x_idx.unsqueeze(-1).repeat((1, 1, 1, 1, 1, 2)))
        hbb = torch.stack([
            min_x_tensors[:, :, :, :, :, 0],
            min_y_tensors[:, :, :, :, :, 1],
            max_x_tensors[:, :, :, :, :, 0],
            max_y_tensors[:, :, :, :, :, 1]
        ], dim=-1)

        return five_params, hbb.reshape((b, -1, 4))

    def process_rpn_proposals(self, rpn_proposals: OrderedDict, anchors: OrderedDict):
        result = OrderedDict()

        # transform proposals to vertices
        for k in rpn_proposals.keys():
            midpoint = anchor_offset_to_midpoint_offset(rpn_proposals[k]["anchor_offsets"], anchors[k])
            vertices = midpoint_offset_to_vertices(midpoint)
            result[k] = vertices

        # parallelogram to rectangular proposals to cv2_format (center_x, center_y, width, height, theta (rad))
        for k, v in result.items():
            b = rpn_proposals[k]['objectness_scores'].shape[0]
            rect_vertices = self.parallelogram_vertices_to_rectangular_vertices(v)
            cv2_format, hbb = self.rectangular_vertices_to_5_param_and_hbb(rect_vertices)
            batch_indexed = []
            vertices = []
            level_scores = []

            for b_idx in range(b):
                # take the top 2000 rpn proposals and apply nms
                topk_k = min(2000, rpn_proposals[k]['objectness_scores'].view((b, -1)).shape[1])
                topk_proposals = torch.topk(rpn_proposals[k]['objectness_scores'][b_idx].reshape(-1), k=topk_k)
                topk_idx = topk_proposals.indices
                topk_scores = topk_proposals.values
                nms_result = nms(hbb[b_idx][topk_idx], topk_scores, 0.5)
                keep = nms_result[1]
                kept_boxes = cv2_format[b_idx, keep]
                b, _, _, h, w = rect_vertices.shape
                flat_vertices = rect_vertices.reshape((b, -1, 4, 2, h, w))
                flat_vertices = torch.movedim(flat_vertices, (2, 3), (4, 5)).reshape(b, -1, 4, 2)
                kept_vertices = flat_vertices[b_idx, keep]
                vertices.append(kept_vertices)
                kept_scores = topk_scores[keep]
                n = kept_boxes.shape[0]
                b_idx_tensor = torch.full((n, 1), b_idx).to(rect_vertices.device)
                values = torch.concatenate((b_idx_tensor, kept_boxes), dim=-1)
                batch_indexed.append(values)
                level_scores.append(kept_scores)

            result[k] = {}
            result[k]["boxes"] = torch.concatenate(batch_indexed, dim=0)
            result[k]["scores"] = torch.stack(level_scores, dim=0)
            result[k]["vertices"] = torch.stack(vertices, dim=0)

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
            # todo: find a better way
            batched_roi_align = {b: [] for b in range(num_batches)}
            batched_boxes = {b: [] for b in range(num_batches)}
            for idx, batch_idx in enumerate(cv2_format[k]["boxes"][:, 0]):
                batched_roi_align[int(batch_idx.item())].append(roi_align[idx])
                batched_boxes[int(batch_idx.item())].append(cv2_format[k]["boxes"][idx])
            merged_features.append(torch.stack([torch.stack(bra, dim=0) for bra in batched_roi_align.values()], dim=0))
            merged_boxes.append(torch.stack([torch.stack(bb, dim=0) for bb in batched_boxes.values()], dim=0) * self.fpn_level_scalings[s_idx])
            merged_scores.append(cv2_format[k]["scores"])
            merged_vertices.append(cv2_format[k]["vertices"])

        merged_features = torch.concatenate(merged_features, dim=1)
        merged_boxes = torch.concatenate(merged_boxes, dim=1)
        merged_scores = torch.concatenate(merged_scores, dim=1)
        merged_vertices = torch.concatenate(merged_vertices, dim=1)

        filtered_features = []
        filtered_boxes = []
        filtered_scores = []
        filtered_vertices = []
        num_batches = merged_features.shape[0]
        for b in range(num_batches):
            keep = torch.topk(merged_scores[b], k=1000).indices
            filtered_features.append(merged_features[b, keep])
            filtered_boxes.append(merged_boxes[b, keep])
            filtered_scores.append(merged_scores[b, keep])
            filtered_vertices.append(merged_vertices[b, keep])

        return {
            "features": torch.stack(filtered_features, dim=0), 
            "boxes": torch.stack(filtered_boxes, dim=0), 
            "scores": torch.stack(filtered_scores, dim=0),
            "vertices": torch.stack(filtered_vertices, dim=0)
        }
