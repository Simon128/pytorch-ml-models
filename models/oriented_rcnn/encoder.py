import torch
from enum import Enum, auto

class Encodings(Enum):
    VERTICES = auto(), # shape (..., 4, 2) [4 vertices á (x,y)]
    ANCHOR_OFFSET = auto(), # shape (..., 6) [δx, δy , δw, δh, δα, δβ]
    MIDPOINT_OFFSET = auto(), # shape(..., 6) [center x, center y, w, h, ∆α, ∆β]
    ORIENTED_CV2_FORMAT = auto(), # shape(..., 5) [center x, center y, w, h, angle in rad]
    HBB_VERTICES = auto(), # shape (..., 4, 2) [4 vertices á (x,y)]
    HBB_CENTERED = auto(), # shape (..., 4) [center x, center y, w, h]
    HBB_CORNERS = auto() # shape (..., 4) [min x, min y, max x, max y]

def encode(
        data: torch.Tensor, 
        src_encoding: Encodings, 
        target_encoding: Encodings,
        anchors: torch.Tensor | None = None 
    ):
    src_encoder = Encoder(data, src_encoding, anchors)
    trgt_encoder = src_encoder.encode(target_encoding)
    return trgt_encoder.get_tensor()

class Encoder:
    def __init__(
            self, 
            data: torch.Tensor, 
            encoding: Encodings,
            anchors: torch.Tensor | None = None 
        ):
        assert encoding in \
            [Encodings.VERTICES, Encodings.ANCHOR_OFFSET, Encodings.MIDPOINT_OFFSET, 
             Encodings.ORIENTED_CV2_FORMAT, Encodings.HBB_CORNERS, Encodings.HBB_VERTICES,
             Encodings.HBB_CENTERED], \
            f"source encoding {encoding} not supported"
        if encoding in [Encodings.VERTICES, Encodings.HBB_VERTICES]:
            assert (
                len(data.shape) >= 2 and data.shape[-1] == 2 and data.shape[-2] == 4
                ), \
                f"data shape ({data.shape}) wrong for encoding {encoding}, expected (..., 4, 2)"
        elif encoding in [Encodings.MIDPOINT_OFFSET, Encodings.ANCHOR_OFFSET]:
            assert (
                len(data.shape) >= 1 and data.shape[-1] == 6
                ), \
                f"data shape ({data.shape}) wrong for encoding {encoding}, expected (..., 6)"
        elif encoding == Encodings.ORIENTED_CV2_FORMAT:
            assert (
                len(data.shape) >= 1 and data.shape[-1] == 5
                ), \
                f"data shape ({data.shape}) wrong for encoding {encoding}, expected (..., 5)"
        elif encoding in [Encodings.HBB_CENTERED, Encodings.HBB_CORNERS]:
            assert (
                len(data.shape) >= 1 and data.shape[-1] == 4
                ), \
                f"data shape ({data.shape}) wrong for encoding {encoding}, expected (..., 4)"
        self.data = data
        self.anchors = anchors
        self.encoding = encoding

    def encode(self, target_encoding: Encodings) -> 'Encoder':
        if target_encoding == Encodings.VERTICES:
            return self.to_vertices()
        elif target_encoding == Encodings.ANCHOR_OFFSET:
            return self.to_anchor_offset()
        elif target_encoding == Encodings.MIDPOINT_OFFSET:
            return self.to_midpoint_offset()
        elif target_encoding == Encodings.ORIENTED_CV2_FORMAT:
            return self.to_oriented_cv2()
        elif target_encoding == Encodings.HBB_VERTICES:
            return self.to_hbb_vertices()
        elif target_encoding == Encodings.HBB_CORNERS:
            return self.to_hbb_corners()
        elif target_encoding == Encodings.HBB_CENTERED:
            return self.to_hbb_centered()
        else:
            raise ValueError(f"target encoding {target_encoding} not supported")

    def to_vertices(self) -> 'Encoder':
        if self.encoding == [Encodings.VERTICES, Encodings.HBB_VERTICES]:
            return self
        elif self.encoding == Encodings.ANCHOR_OFFSET:
            assert self.anchors is not None, f"anchors required for encoding {self.encoding} -> {Encodings.VERTICES}"
            midpoint_offset = self.__anchor_offset_to_midpoint_offset(self.data, self.anchors)
            vertices = self.__midpoint_offset_to_vertices(midpoint_offset)
        elif self.encoding == Encodings.MIDPOINT_OFFSET:
            vertices = self.__midpoint_offset_to_vertices(self.data)
        elif self.encoding == Encodings.ORIENTED_CV2_FORMAT:
            vertices = self.__oriented_cv2_to_vertices(self.data)
        elif self.encoding == Encodings.HBB_CENTERED:
            vertices = self.__hbb_centered_to_hbb_vertices(self.data)
        elif self.encoding == Encodings.HBB_CORNERS:
            vertices = self.__hbb_corners_to_hbb_vertices(self.data)
        else:
            raise ValueError(f"source encoding {self.encoding} not supported")

        return Encoder(vertices, Encodings.VERTICES, self.anchors)

    def to_anchor_offset(self) -> 'Encoder':
        assert self.anchors is not None, f"anchors required for encoding {self.encoding} -> {Encodings.ANCHOR_OFFSET}"
        if self.encoding in [Encodings.VERTICES, Encodings.HBB_VERTICES]:
            midpoint_offset = self.__vertices_to_midpoint_offset(self.data)
            anchor_offset = self.__midpoint_offset_to_anchor_offset(midpoint_offset, self.anchors)
        elif self.encoding == Encodings.ANCHOR_OFFSET:
            return self
        elif self.encoding == Encodings.MIDPOINT_OFFSET:
            anchor_offset = self.__midpoint_offset_to_anchor_offset(self.data, self.anchors)
        elif self.encoding == Encodings.ORIENTED_CV2_FORMAT:
            vertices = self.__oriented_cv2_to_vertices(self.data)
            midpoint_offset = self.__vertices_to_midpoint_offset(vertices)
            anchor_offset = self.__midpoint_offset_to_anchor_offset(midpoint_offset, self.anchors)
        elif self.encoding == Encodings.HBB_CENTERED:
            vertices = self.__hbb_centered_to_hbb_vertices(self.data)
            midpoint_offset = self.__vertices_to_midpoint_offset(vertices)
            anchor_offset = self.__midpoint_offset_to_anchor_offset(midpoint_offset, self.anchors)
        elif self.encoding == Encodings.HBB_CORNERS:
            vertices = self.__hbb_corners_to_hbb_vertices(self.data)
            midpoint_offset = self.__vertices_to_midpoint_offset(vertices)
            anchor_offset = self.__midpoint_offset_to_anchor_offset(midpoint_offset, self.anchors)
        else:
            raise ValueError(f"source encoding {self.encoding} not supported")

        return Encoder(anchor_offset, Encodings.ANCHOR_OFFSET, self.anchors)

    def to_midpoint_offset(self) -> 'Encoder':
        if self.encoding in [Encodings.VERTICES, Encodings.HBB_VERTICES]:
            midpoint_offset = self.__vertices_to_midpoint_offset(self.data)
        elif self.encoding == Encodings.ANCHOR_OFFSET:
            assert self.anchors is not None, f"anchors required for encoding {self.encoding} -> {Encodings.MIDPOINT_OFFSET}"
            midpoint_offset = self.__anchor_offset_to_midpoint_offset(self.data, self.anchors)
        elif self.encoding == Encodings.MIDPOINT_OFFSET:
            return self
        elif self.encoding == Encodings.ORIENTED_CV2_FORMAT:
            vertices = self.__oriented_cv2_to_vertices(self.data)
            midpoint_offset = self.__vertices_to_midpoint_offset(vertices)
        elif self.encoding == Encodings.HBB_CENTERED:
            vertices = self.__hbb_centered_to_hbb_vertices(self.data)
            midpoint_offset = self.__vertices_to_midpoint_offset(vertices)
        elif self.encoding == Encodings.HBB_CORNERS:
            vertices = self.__hbb_corners_to_hbb_vertices(self.data)
            midpoint_offset = self.__vertices_to_midpoint_offset(vertices)
        else:
            raise ValueError(f"source encoding {self.encoding} not supported")

        return Encoder(midpoint_offset, Encodings.MIDPOINT_OFFSET, self.anchors)

    def to_oriented_cv2(self) -> 'Encoder':
        if self.encoding in [Encodings.VERTICES, Encodings.HBB_VERTICES]:
            oriented_cv2 = self.__vertices_to_oriented_cv2(self.data)
        elif self.encoding == Encodings.ANCHOR_OFFSET:
            assert self.anchors is not None, f"anchors required for encoding {self.encoding} -> {Encodings.ORIENTED_CV2_FORMAT}"
            midpoint_offset = self.__anchor_offset_to_midpoint_offset(self.data, self.anchors)
            vertices = self.__midpoint_offset_to_vertices(midpoint_offset)
            oriented_cv2 = self.__vertices_to_oriented_cv2(vertices)
        elif self.encoding == Encodings.MIDPOINT_OFFSET:
            vertices = self.__midpoint_offset_to_vertices(self.data)
            oriented_cv2 = self.__vertices_to_oriented_cv2(vertices)
        elif self.encoding == Encodings.ORIENTED_CV2_FORMAT:
            return self
        elif self.encoding == Encodings.HBB_CENTERED:
            vertices = self.__hbb_centered_to_hbb_vertices(self.data)
            oriented_cv2 = self.__vertices_to_oriented_cv2(vertices)
        elif self.encoding == Encodings.HBB_CORNERS:
            vertices = self.__hbb_corners_to_hbb_vertices(self.data)
            oriented_cv2 = self.__vertices_to_oriented_cv2(vertices)
        else:
            raise ValueError(f"source encoding {self.encoding} not supported")

        return Encoder(oriented_cv2, Encodings.ORIENTED_CV2_FORMAT, self.anchors)

    def to_hbb_vertices(self) -> 'Encoder':
        if self.encoding == Encodings.VERTICES:
            hbb_vertices = self.__vertices_to_hbb_vertices(self.data)
        elif self.encoding == Encodings.HBB_VERTICES:
            return self
        elif self.encoding == Encodings.ANCHOR_OFFSET:
            assert self.anchors is not None, f"anchors required for encoding {self.encoding} -> {Encodings.ORIENTED_CV2_FORMAT}"
            midpoint_offset = self.__anchor_offset_to_midpoint_offset(self.data, self.anchors)
            vertices = self.__midpoint_offset_to_vertices(midpoint_offset)
            hbb_vertices = self.__vertices_to_hbb_vertices(vertices)
        elif self.encoding == Encodings.MIDPOINT_OFFSET:
            vertices = self.__midpoint_offset_to_vertices(self.data)
            hbb_vertices = self.__vertices_to_hbb_vertices(vertices)
        elif self.encoding == Encodings.ORIENTED_CV2_FORMAT:
            vertices = self.__oriented_cv2_to_vertices(self.data)
            hbb_vertices = self.__vertices_to_hbb_vertices(vertices)
        elif self.encoding == Encodings.HBB_CENTERED:
            hbb_vertices = self.__hbb_centered_to_hbb_vertices(self.data)
        elif self.encoding == Encodings.HBB_CORNERS:
            hbb_vertices = self.__hbb_corners_to_hbb_vertices(self.data)
        else:
            raise ValueError(f"source encoding {self.encoding} not supported")

        return Encoder(hbb_vertices, Encodings.HBB_VERTICES, self.anchors)

    def to_hbb_corners(self) -> 'Encoder':
        if self.encoding == Encodings.VERTICES:
            hbb_vertices = self.__vertices_to_hbb_vertices(self.data)
            hbb_corners = self.__hbb_vertices_to_hbb_corners(hbb_vertices)
        elif self.encoding == Encodings.HBB_VERTICES:
            hbb_corners = self.__hbb_vertices_to_hbb_corners(self.data)
        elif self.encoding == Encodings.ANCHOR_OFFSET:
            assert self.anchors is not None, f"anchors required for encoding {self.encoding} -> {Encodings.ORIENTED_CV2_FORMAT}"
            midpoint_offset = self.__anchor_offset_to_midpoint_offset(self.data, self.anchors)
            vertices = self.__midpoint_offset_to_vertices(midpoint_offset)
            hbb_vertices = self.__vertices_to_hbb_vertices(vertices)
            hbb_corners = self.__hbb_vertices_to_hbb_corners(hbb_vertices)
        elif self.encoding == Encodings.MIDPOINT_OFFSET:
            vertices = self.__midpoint_offset_to_vertices(self.data)
            hbb_vertices = self.__vertices_to_hbb_vertices(vertices)
            hbb_corners = self.__hbb_vertices_to_hbb_corners(hbb_vertices)
        elif self.encoding == Encodings.ORIENTED_CV2_FORMAT:
            vertices = self.__oriented_cv2_to_vertices(self.data)
            hbb_vertices = self.__vertices_to_hbb_vertices(vertices)
            hbb_corners = self.__hbb_vertices_to_hbb_corners(hbb_vertices)
        elif self.encoding == Encodings.HBB_CENTERED:
            hbb_vertices = self.__hbb_centered_to_hbb_vertices(self.data)
            hbb_corners = self.__hbb_vertices_to_hbb_corners(hbb_vertices)
        elif self.encoding == Encodings.HBB_CORNERS:
            return self
        else:
            raise ValueError(f"source encoding {self.encoding} not supported")

        return Encoder(hbb_corners, Encodings.HBB_CORNERS, self.anchors)

    def to_hbb_centered(self) -> 'Encoder':
        if self.encoding == Encodings.VERTICES:
            hbb_vertices = self.__vertices_to_hbb_vertices(self.data)
            hbb_centered = self.__hbb_vertices_to_hbb_centered(hbb_vertices)
        elif self.encoding == Encodings.HBB_VERTICES:
            hbb_centered = self.__hbb_vertices_to_hbb_centered(self.data)
        elif self.encoding == Encodings.ANCHOR_OFFSET:
            assert self.anchors is not None, f"anchors required for encoding {self.encoding} -> {Encodings.ORIENTED_CV2_FORMAT}"
            midpoint_offset = self.__anchor_offset_to_midpoint_offset(self.data, self.anchors)
            vertices = self.__midpoint_offset_to_vertices(midpoint_offset)
            hbb_vertices = self.__vertices_to_hbb_vertices(vertices)
            hbb_centered = self.__hbb_vertices_to_hbb_centered(hbb_vertices)
        elif self.encoding == Encodings.MIDPOINT_OFFSET:
            vertices = self.__midpoint_offset_to_vertices(self.data)
            hbb_vertices = self.__vertices_to_hbb_vertices(vertices)
            hbb_centered = self.__hbb_vertices_to_hbb_centered(hbb_vertices)
        elif self.encoding == Encodings.ORIENTED_CV2_FORMAT:
            vertices = self.__oriented_cv2_to_vertices(self.data)
            hbb_vertices = self.__vertices_to_hbb_vertices(vertices)
            hbb_centered = self.__hbb_vertices_to_hbb_centered(hbb_vertices)
        elif self.encoding == Encodings.HBB_CENTERED:
            return self
        elif self.encoding == Encodings.HBB_CORNERS:
            hbb_vertices = self.__hbb_corners_to_hbb_vertices(self.data)
            hbb_centered = self.__hbb_vertices_to_hbb_centered(hbb_vertices)
        else:
            raise ValueError(f"source encoding {self.encoding} not supported")

        return Encoder(hbb_centered, Encodings.HBB_CENTERED, self.anchors)

    def get_tensor(self) -> torch.Tensor:
        return self.data

    def __anchor_offset_to_midpoint_offset(self, anchor_offset: torch.Tensor, anchors: torch.Tensor):
        w = anchors[..., 2] * torch.exp(anchor_offset[..., 2])
        h = anchors[..., 3] * torch.exp(anchor_offset[..., 3])
        x = anchor_offset[..., 0] * anchors[..., 2] + anchors[..., 0]
        y = anchor_offset[..., 1] * anchors[..., 3] + anchors[..., 1]
        delta_alpha = anchor_offset[..., 4] * w
        delta_beta = anchor_offset[..., 5] * h
        midpoint_offset = torch.stack((x, y, w, h, delta_alpha, delta_beta), dim=-1)
        return midpoint_offset.float()

    def __midpoint_offset_to_anchor_offset(self, midpoint_offset: torch.Tensor, anchors: torch.Tensor):
        d_a = midpoint_offset[..., 4] / midpoint_offset[..., 2]
        d_b = midpoint_offset[..., 5] / midpoint_offset[..., 3]
        d_w = torch.log(midpoint_offset[..., 2] / anchors[..., 2])
        d_h = torch.log(midpoint_offset[..., 3] / anchors[..., 3])
        d_x = (midpoint_offset[..., 0] - anchors[..., 0]) / anchors[..., 2]
        d_y = (midpoint_offset[..., 1] - anchors[..., 1]) / anchors[..., 3]
        return torch.stack((d_x, d_y, d_w, d_h, d_a, d_b), dim=-1).float()

    def __midpoint_offset_to_vertices(self, midpoint_offset: torch.Tensor):
        x = midpoint_offset[..., 0]
        y = midpoint_offset[..., 1]
        w = midpoint_offset[..., 2]
        h = midpoint_offset[..., 3]
        d_alpha = midpoint_offset[..., 4]
        d_beta = midpoint_offset[..., 5]
        v1 = torch.stack([x + d_alpha, y - h / 2], dim=-1)
        v2 = torch.stack([x + w / 2, y + d_beta], dim=-1)
        v3 = torch.stack([x - d_alpha, y + h / 2], dim=-1)
        v4 = torch.stack([x - w / 2, y - d_beta], dim=-1)
        return torch.stack((v1, v2, v3, v4), dim=-2).float()

    def __vertices_to_midpoint_offset(self, vertices: torch.Tensor):
        x_min = torch.min(vertices[..., 0], dim=-1)[0]
        x_max = torch.max(vertices[..., 0], dim=-1)[0]
        y_min = torch.min(vertices[..., 1], dim=-1)[0]
        y_max = torch.max(vertices[..., 1], dim=-1)[0]
        w = x_max - x_min
        h = y_max - y_min
        x_center = x_min + w / 2
        y_center = y_min + h / 2
        delta_a = vertices[..., 0, 0] - x_center
        delta_b = vertices[..., 1, 1] - y_center
        return torch.stack((x_center, y_center, w, h, delta_a, delta_b), dim=-1)

    def __vertices_to_oriented_cv2(self, vertices: torch.Tensor):
        # transform rectangular vertices to (x, y, w, h, theta)
        # with x,y being center coordinates of box and theta 
        # correponding to the theta as defined by the mmcv RoiAlignRotated 
        # clockwise assumption
        # (first min_y will be the left one if there are two)
        repeat_list = [1] * len(vertices.shape[:-1])
        repeat_list.append(2)
        repeat = tuple(repeat_list)
        min_y_idx = torch.argmin(vertices[..., 1], dim=-1, keepdim=True)
        min_y_tensors = torch.gather(vertices, -2, min_y_idx.unsqueeze(-1).repeat(repeat))
        # for the reference vector, we need the correct neighbouring vertex 
        # which is the one with largest x coord
        max_x_idx = torch.argmax(vertices[..., 0], dim=-1, keepdim=True)
        max_x_tensors = torch.gather(vertices, -2, max_x_idx.unsqueeze(-1).repeat(repeat))
        ref_vector = max_x_tensors - min_y_tensors
        angle = torch.arccos(ref_vector[..., 0] / (torch.norm(ref_vector, dim=-1) + 1))
        width = max_x_tensors[..., 0] - min_y_tensors[..., 0]
        x_center = min_y_tensors[..., 0] + width/2
        max_y_idx = torch.argmax(vertices[..., 1], dim=-1, keepdim=True)
        max_y_tensors = torch.gather(vertices, -2, max_y_idx.unsqueeze(-1).repeat(repeat))
        height =  max_y_tensors[..., 1] - min_y_tensors[..., 1]
        y_center = min_y_tensors[..., 1] + height / 2
        five_params = torch.stack((x_center, y_center, width, height, angle), dim=-1).reshape((-1, 5))
        return five_params

    def __oriented_cv2_to_vertices(self, vertices: torch.Tensor):
        raise NotImplementedError("do we need this?")

    def __hbb_centered_to_hbb_vertices(self, hbb_centered: torch.Tensor):
        center_x = hbb_centered[..., 0]
        center_y = hbb_centered[..., 1]
        width = hbb_centered[..., 2]
        height = hbb_centered[..., 3]
        v1 = torch.stack((center_x - width / 2, center_y - height / 2), dim=-1)
        v2 = torch.stack((center_x + width / 2, center_y - height / 2), dim=-1)
        v3 = torch.stack((center_x + width / 2, center_y + height / 2), dim=-1)
        v4 = torch.stack((center_x - width / 2, center_y + height / 2), dim=-1)
        return torch.stack((v1, v2, v3, v4), dim=-2)

    def __hbb_vertices_to_hbb_centered(self, hbb_vertices: torch.Tensor):
        min_x = torch.min(hbb_vertices[..., 0], dim=-1).values
        min_y = torch.min(hbb_vertices[..., 1], dim=-1).values
        max_x = torch.max(hbb_vertices[..., 0], dim=-1).values
        max_y = torch.max(hbb_vertices[..., 1], dim=-1).values
        width = max_x - min_x
        height = max_y - min_y
        center_x = min_x + width / 2
        center_y = min_y + height / 2
        return torch.stack((center_x, center_y, width, height), dim=-1)

    def __hbb_corners_to_hbb_vertices(self, hbb_corners: torch.Tensor):
        min_x = hbb_corners[..., 0]
        min_y = hbb_corners[..., 1]
        max_x = hbb_corners[..., 2]
        max_y = hbb_corners[..., 3]
        v1 = torch.stack((min_x, min_y), dim=-1)
        v2 = torch.stack((max_x, min_y), dim=-1)
        v3 = torch.stack((max_x, max_y), dim=-1)
        v4 = torch.stack((min_x, max_y), dim=-1)
        return torch.stack((v1, v2, v3, v4), dim=-2)

    def __hbb_vertices_to_hbb_corners(self, hbb_vertices: torch.Tensor):
        min_x = torch.min(hbb_vertices[..., 0], dim=-1).values
        min_y = torch.min(hbb_vertices[..., 1], dim=-1).values
        max_x = torch.max(hbb_vertices[..., 0], dim=-1).values
        max_y = torch.max(hbb_vertices[..., 1], dim=-1).values
        return torch.stack((min_x, min_y, max_x, max_y), dim=-1)

    def __vertices_to_hbb_vertices(self, vertices: torch.Tensor):
        min_x = torch.min(vertices[..., 0], dim=-1).values
        min_y = torch.min(vertices[..., 1], dim=-1).values
        max_x = torch.max(vertices[..., 0], dim=-1).values
        max_y = torch.max(vertices[..., 1], dim=-1).values
        v1 = torch.stack((min_x, min_y), dim=-1)
        v2 = torch.stack((max_x, min_y), dim=-1)
        v3 = torch.stack((max_x, max_y), dim=-1)
        v4 = torch.stack((min_x, max_y), dim=-1)
        return torch.stack((v1, v2, v3, v4), dim=-2)
