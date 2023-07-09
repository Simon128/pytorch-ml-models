from mmcv.ops import RoIAlignRotated
import torch
from collections import OrderedDict

from ..encodings import anchor_offset_to_midpoint_offset, midpoint_offset_to_vertices

class RoIAlignRotatedWrapper(RoIAlignRotated):
    def __init__(self, output_size, spatial_scale, sampling_ratio: int):
        super().__init__(output_size, spatial_scale, sampling_ratio=sampling_ratio)

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
        )
        diag2_len = torch.sqrt(
            torch.square(v4[:, :, 0, :, :] - v2[:, :, 0, :, :]) + 
            torch.square(v4[:, :, 1, :, :] - v2[:, :, 1, :, :])
        )
    
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
        return torch.cat([r_rectangular[:, i, :, :, :, :] for i in range(num_rois)])

    def rectangular_vertices_to_5_param(self, rect_v: torch.Tensor):
        # transform rectangular vertices to (x, y, w, h, theta)
        # with x,y being center coordinates of box and theta 
        # correponding to the theta as defined by the mmcv RoiAlignRotated 
        b, n, _, h, w = rect_v.shape
        num_rois = int(n/4)
        r_rect = rect_v.reshape((b, num_rois, 4, 2, h, w))
        # clockwise assumption
        # (first min_y will be the left one if there are two)
        min_y_idx = torch.argmin(rect_v[:, :, :, 1, :, :], dim=2)
        # for the reference vector, we need the correct neighbouring vertex 
        # which is the one with largest x coord
        max_x_vals, max_x_idx = torch.max(rect_v[:, :, :, 0, :, :], dim=2)
        ref_vector = rect_v[:, :, max_x_idx] - rect_v[:, :, min_y_idx, :, :, :]

        # angle between two vectors: cos(theta) = (a * b) / (|a| + |b|)
        # todo: 1. find reference vertex (https://github.com/open-mmlab/mmrotate/blob/main/docs/en/intro.md)
        # 2. compute vectors of both sides from the vertex (can we maybe preselect the one with smaller angle?)
        # 3. compute pairwise angle between vectors and x-axis (starting from reference point)
        # 4. select smaller angle (if 0° -> 90°)
        pass


    def process_rpn_proposals(self, rpn_proposals: OrderedDict, anchors: OrderedDict):
        result = OrderedDict()

        # transform proposals to vertices
        for k in rpn_proposals.keys():
            midpoint = anchor_offset_to_midpoint_offset(rpn_proposals[k], anchors[k])
            vertices = midpoint_offset_to_vertices(midpoint)
            result[k] = vertices

        # parallelogram to rectangular proposals to cv2_format (center_x, center_y, width, height, theta (rad))
        for k, v in result.items():
            rect_vertices = self.parallelogram_vertices_to_rectangular_vertices(v)
            cv2_format = self.rectangular_vertices_to_5_param(rect_vertices)
            result[k] = cv2_format

        return result

    def forward(self, fpn_features: OrderedDict, rpn_proposals: OrderedDict, anchors: OrderedDict):
        cv2_format = self.process_rpn_proposals(rpn_proposals, anchors)
        result = OrderedDict()

        for k in cv2_format.keys():
            result[k] = super().forward(fpn_features[k], cv2_format[k])

        return result
