import torch
from enum import Enum, auto
import numpy as np

EPS = 1e-7

class Encodings(Enum):
    VERTICES = auto(), # shape (..., 4, 2) [4 vertices á (x,y)]
    ANCHOR_OFFSET = auto(), # shape (..., 6) [δx, δy , δw, δh, δα, δβ]
    MIDPOINT_OFFSET = auto(), # shape(..., 6) [center x, center y, w, h, ∆α, ∆β]
    THETA_FORMAT_TL_RT = auto() 
    # shape(..., 5) [center x, center y, w, h, angle in rad]  ref vector for angle: Top left -> Top right
    THETA_FORMAT_BL_RB = auto()  
    # shape(..., 5) [center x, center y, w, h, angle in rad]  ref vector for angle: Bot left -> Bot right
    HBB_VERTICES = auto(), # shape (..., 4, 2) [4 vertices á (x,y)]
    HBB_CENTERED = auto(), # shape (..., 4) [center x, center y, w, h]
    HBB_CORNERS = auto() # shape (..., 4) [min x, min y, max x, max y]

def encode(
        data: torch.Tensor, 
        src_encoding: Encodings, 
        target_encoding: Encodings,
        anchors: torch.Tensor | None = None,
     
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
             Encodings.THETA_FORMAT_TL_RT, Encodings.HBB_CORNERS, Encodings.HBB_VERTICES,
             Encodings.HBB_CENTERED, Encodings.THETA_FORMAT_BL_RB], \
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
            data = torch.clamp(data, min=-100000, max=100000)
        elif encoding in [Encodings.THETA_FORMAT_BL_RB, Encodings.THETA_FORMAT_TL_RT]:
            assert (
                len(data.shape) >= 1 and data.shape[-1] == 5
                ), \
                f"data shape ({data.shape}) wrong for encoding {encoding}, expected (..., 5)"
        elif encoding in [Encodings.HBB_CENTERED, Encodings.HBB_CORNERS]:
            assert (
                len(data.shape) >= 1 and data.shape[-1] == 4
                ), \
                f"data shape ({data.shape}) wrong for encoding {encoding}, expected (..., 4)"
        # prevent malformed data
        self.data = torch.clamp(data, min=-100000, max=100000)
        self.anchors = anchors
        self.encoding = encoding

    def encode(self, target_encoding: Encodings) -> 'Encoder':
        if target_encoding == Encodings.VERTICES:
            return self.to_vertices()
        elif target_encoding == Encodings.ANCHOR_OFFSET:
            return self.to_anchor_offset()
        elif target_encoding == Encodings.MIDPOINT_OFFSET:
            return self.to_midpoint_offset()
        elif target_encoding in [Encodings.THETA_FORMAT_TL_RT, Encodings.THETA_FORMAT_BL_RB]:
            return self.to_theta_format(target_encoding)
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
        elif self.encoding in [Encodings.THETA_FORMAT_BL_RB, Encodings.THETA_FORMAT_TL_RT]:
            vertices = self.__theta_to_vertices(self.data, self.encoding)
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
        elif self.encoding in [Encodings.THETA_FORMAT_BL_RB, Encodings.THETA_FORMAT_TL_RT]:
            vertices = self.__theta_to_vertices(self.data, self.encoding)
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
        elif self.encoding in [Encodings.THETA_FORMAT_BL_RB, Encodings.THETA_FORMAT_TL_RT]:
            vertices = self.__theta_to_vertices(self.data, self.encoding)
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

    def to_theta_format(self, specific: Encodings) -> 'Encoder':
        if self.encoding in [Encodings.VERTICES, Encodings.HBB_VERTICES]:
            theta_format = self.__vertices_to_theta(self.data, specific)
        elif self.encoding == Encodings.ANCHOR_OFFSET:
            assert self.anchors is not None, f"anchors required for encoding {self.encoding} -> {Encodings.ORIENTED_CV2_FORMAT}"
            midpoint_offset = self.__anchor_offset_to_midpoint_offset(self.data, self.anchors)
            vertices = self.__midpoint_offset_to_vertices(midpoint_offset)
            theta_format = self.__vertices_to_theta(vertices, specific)
        elif self.encoding == Encodings.MIDPOINT_OFFSET:
            vertices = self.__midpoint_offset_to_vertices(self.data)
            theta_format = self.__vertices_to_theta(vertices, specific)
        elif self.encoding in [Encodings.THETA_FORMAT_BL_RB, Encodings.THETA_FORMAT_TL_RT]:
            vertices = self.__theta_to_vertices(self.data, self.encoding)
            theta_format = self.__vertices_to_theta(vertices, specific)
        elif self.encoding == Encodings.HBB_CENTERED:
            vertices = self.__hbb_centered_to_hbb_vertices(self.data)
            theta_format = self.__vertices_to_theta(vertices, specific)
        elif self.encoding == Encodings.HBB_CORNERS:
            vertices = self.__hbb_corners_to_hbb_vertices(self.data)
            theta_format = self.__vertices_to_theta(vertices, specific)
        else:
            raise ValueError(f"source encoding {self.encoding} not supported")

        return Encoder(theta_format, specific, self.anchors)

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
        elif self.encoding in [Encodings.THETA_FORMAT_BL_RB, Encodings.THETA_FORMAT_TL_RT]:
            hbb_vertices = self.__hbb_centered_to_hbb_vertices(self.data[..., :-1])
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
        elif self.encoding in [Encodings.THETA_FORMAT_BL_RB, Encodings.THETA_FORMAT_TL_RT]:
            hbb_vertices = self.__hbb_centered_to_hbb_vertices(self.data[..., :-1])
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
        elif self.encoding in [Encodings.THETA_FORMAT_BL_RB, Encodings.THETA_FORMAT_TL_RT]:
            vertices = self.__theta_to_vertices(self.data)
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
        # we have to clamp the anchor offsets for the exp functions
        # to prevent inf and therefore nan gradients
        w = anchors[..., 2] * torch.exp(torch.clamp(anchor_offset[..., 2], min=-10, max=10))
        h = anchors[..., 3] * torch.exp(torch.clamp(anchor_offset[..., 3], min=-10, max=10))
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
        tl = self.__get_top_left_vertices(vertices)
        rt = self.__get_right_top_vertices(vertices)
        x_min = torch.min(vertices[..., 0], dim=-1)[0]
        x_max = torch.max(vertices[..., 0], dim=-1)[0]
        y_min = torch.min(vertices[..., 1], dim=-1)[0]
        y_max = torch.max(vertices[..., 1], dim=-1)[0]
        w = x_max - x_min
        h = y_max - y_min
        x_center = x_min + w / 2
        y_center = y_min + h / 2
        delta_a = tl[..., 0, 0] - x_center
        delta_b = rt[..., 0, 1] - y_center
        return torch.stack((x_center, y_center, w, h, delta_a, delta_b), dim=-1)

    def __get_top_left_vertices(self, vertices: torch.Tensor):
        repeat_list = [1] * len(vertices.shape[:-1])
        repeat_list.append(2)
        repeat = tuple(repeat_list)
        # torch argmin returns the first minimal vertex
        # this is fine, if the angle is not pi/2 (axis aligned)
        min_y_idx = torch.argmin(vertices[..., 1], dim=-1, keepdim=True)
        min_y_tensors = torch.gather(vertices, -2, min_y_idx.unsqueeze(-1).repeat(repeat))

        # however, if the angle is pi/2, then we have two min_y vertices
        # we need to check, which on is the left one (smaller x)
        # for this, we just flip the order of the vertices and use argmin
        flipped_vertices = torch.flip(vertices, dims=(-2,))
        min_y_idx_second = torch.argmin(flipped_vertices[..., 1], dim=-1, keepdim=True)
        min_y_tensors_second = torch.gather(flipped_vertices, -2, min_y_idx_second.unsqueeze(-1).repeat(repeat))

        # now we just take the vertices with smaller x
        tensors = torch.where(
            min_y_tensors[..., 0].unsqueeze(-1).repeat(repeat) > min_y_tensors_second[..., 0].unsqueeze(-1).repeat(repeat), 
            min_y_tensors_second, 
            min_y_tensors
        )
        return tensors
    
    def __get_right_top_vertices(self, vertices: torch.Tensor):
        repeat_list = [1] * len(vertices.shape[:-1])
        repeat_list.append(2)
        repeat = tuple(repeat_list)
        # torch argmax returns the first maximal vertex
        # this is fine, if the angle is not pi/2 (axis aligned)
        max_x_idx = torch.argmax(vertices[..., 0], dim=-1, keepdim=True)
        max_x_tensors = torch.gather(vertices, -2, max_x_idx.unsqueeze(-1).repeat(repeat))

        # however, if the angle is pi/2, then we have two max x vertices
        # we need to check, which on is the left one (smaller y)
        # for this, we just flip the order of the vertices and use argmax
        flipped_vertices = torch.flip(vertices, dims=(-2,))
        max_x_idx_second = torch.argmax(flipped_vertices[..., 0], dim=-1, keepdim=True)
        max_x_tensors_second = torch.gather(flipped_vertices, -2, max_x_idx_second.unsqueeze(-1).repeat(repeat))

        # now we just take the vertices with smaller y 
        tensors = torch.where(
            max_x_tensors[..., 1].unsqueeze(-1).repeat(repeat) > max_x_tensors_second[..., 1].unsqueeze(-1).repeat(repeat), 
            max_x_tensors_second, 
            max_x_tensors
        )
        return tensors

    def __get_left_bot_vertices(self, vertices: torch.Tensor):
        repeat_list = [1] * len(vertices.shape[:-1])
        repeat_list.append(2)
        repeat = tuple(repeat_list)
        # torch argmin returns the first minial vertex
        # this is fine, if the angle is not pi/2 (axis aligned)
        min_x_idx = torch.argmin(vertices[..., 0], dim=-1, keepdim=True)
        min_x_tensors = torch.gather(vertices, -2, min_x_idx.unsqueeze(-1).repeat(repeat))

        # however, if the angle is pi/2, then we have two min x vertices
        # we need to check, which on is the bot one (larger y)
        # for this, we just flip the order of the vertices and use argmax
        flipped_vertices = torch.flip(vertices, dims=(-2,))
        min_x_idx_second = torch.argmin(flipped_vertices[..., 0], dim=-1, keepdim=True)
        min_x_tensors_second = torch.gather(flipped_vertices, -2, min_x_idx_second.unsqueeze(-1).repeat(repeat))

        # now we just take the vertices with larger y 
        tensors = torch.where(
            min_x_tensors[..., 1].unsqueeze(-1).repeat(repeat) < min_x_tensors_second[..., 1].unsqueeze(-1).repeat(repeat), 
            min_x_tensors_second, 
            min_x_tensors
        )
        return tensors

    def __get_left_top_vertices(self, vertices: torch.Tensor):
        repeat_list = [1] * len(vertices.shape[:-1])
        repeat_list.append(2)
        repeat = tuple(repeat_list)
        # torch argmin returns the first minial vertex
        # this is fine, if the angle is not pi/2 (axis aligned)
        min_x_idx = torch.argmin(vertices[..., 0], dim=-1, keepdim=True)
        min_x_tensors = torch.gather(vertices, -2, min_x_idx.unsqueeze(-1).repeat(repeat))

        # however, if the angle is pi/2, then we have two min x vertices
        # we need to check, which on is the bot one (larger y)
        # for this, we just flip the order of the vertices and use argmax
        flipped_vertices = torch.flip(vertices, dims=(-2,))
        min_x_idx_second = torch.argmin(flipped_vertices[..., 0], dim=-1, keepdim=True)
        min_x_tensors_second = torch.gather(flipped_vertices, -2, min_x_idx_second.unsqueeze(-1).repeat(repeat))

        # now we just take the vertices with larger y 
        tensors = torch.where(
            min_x_tensors[..., 1].unsqueeze(-1).repeat(repeat) > min_x_tensors_second[..., 1].unsqueeze(-1).repeat(repeat), 
            min_x_tensors_second, 
            min_x_tensors
        )
        return tensors

    def __get_bot_left_vertices(self, vertices: torch.Tensor):
        repeat_list = [1] * len(vertices.shape[:-1])
        repeat_list.append(2)
        repeat = tuple(repeat_list)
        # torch argmin returns the first minial vertex
        # this is fine, if the angle is not pi/2 (axis aligned)
        min_y_idx = torch.argmax(vertices[..., 1], dim=-1, keepdim=True)
        min_y_tensors = torch.gather(vertices, -2, min_y_idx.unsqueeze(-1).repeat(repeat))

        # however, if the angle is pi/2, then we have two min x vertices
        # we need to check, which on is the bot one (larger y)
        # for this, we just flip the order of the vertices and use argmax
        flipped_vertices = torch.flip(vertices, dims=(-2,))
        min_y_idx_second = torch.argmax(flipped_vertices[..., 1], dim=-1, keepdim=True)
        min_y_tensors_second = torch.gather(flipped_vertices, -2, min_y_idx_second.unsqueeze(-1).repeat(repeat))

        # now we just take the vertices with larger y 
        tensors = torch.where(
            min_y_tensors[..., 0].unsqueeze(-1).repeat(repeat) > min_y_tensors_second[..., 0].unsqueeze(-1).repeat(repeat), 
            min_y_tensors_second, 
            min_y_tensors
        )
        return tensors

    def __get_right_bot_vertices(self, vertices: torch.Tensor):
        repeat_list = [1] * len(vertices.shape[:-1])
        repeat_list.append(2)
        repeat = tuple(repeat_list)
        # torch argmin returns the first minial vertex
        # this is fine, if the angle is not pi/2 (axis aligned)
        max_x_idx = torch.argmax(vertices[..., 0], dim=-1, keepdim=True)
        max_x_tensors = torch.gather(vertices, -2, max_x_idx.unsqueeze(-1).repeat(repeat))

        # however, if the angle is pi/2, then we have two max y vertices
        # we need to check, which one is the right one (larger x)
        # for this, we just flip the order of the vertices and use argmax
        flipped_vertices = torch.flip(vertices, dims=(-2,))
        max_x_idx_second = torch.argmax(flipped_vertices[..., 0], dim=-1, keepdim=True)
        max_x_tensors_second = torch.gather(flipped_vertices, -2, max_x_idx_second.unsqueeze(-1).repeat(repeat))

        # now we just take the vertices with larger x 
        tensors = torch.where(
            max_x_tensors[..., 1].unsqueeze(-1).repeat(repeat) > max_x_tensors_second[..., 1].unsqueeze(-1).repeat(repeat), 
            max_x_tensors, 
            max_x_tensors_second
        )
        return tensors

    def __parallelogram_vertices_to_rectangular_vertices(self, parallelogram: torch.Tensor):
        # we get the vectors of both diagonales,
        # normalize them by length
        # and for the shorter diagonal we add the corresponding norm. vector
        # to both endpoints (vertices) scaled by the diag. length difference / 2
        rep = [1] * (len(parallelogram.shape) - 1)
        rep[-1] += 1
        rep = tuple(rep)

        v1 = parallelogram[..., 0, :]
        v2 = parallelogram[..., 1, :]
        v3 = parallelogram[..., 2, :]
        v4 = parallelogram[..., 3, :]
        # see https://github.com/pytorch/pytorch/issues/43211
        diag1_len = torch.norm(v3 - v1, dim=-1).unsqueeze(-1).repeat(rep)
        diag2_len = torch.norm(v4 - v2, dim=-1).unsqueeze(-1).repeat(rep)
    
        # assume diag1_len > diag2_len
        # extend diag2
        extension_len = (diag1_len - diag2_len) / 2
        norm_ext_vector = (v4 - v2) / torch.clamp(diag2_len, EPS)
        new_v4 = v4 + norm_ext_vector * extension_len
        new_v2 = v2 + -1 * norm_ext_vector * extension_len

        # assume diag1_len < diag2_len
        # extend diag1
        extension_len = (diag2_len - diag1_len) / 2
        norm_ext_vector = (v3 - v1) / torch.clamp(diag1_len, EPS)
        new_v3 = v3 + norm_ext_vector * extension_len
        new_v1 = v1 + -1 * norm_ext_vector * extension_len

        v1_new = torch.where(diag1_len > diag2_len, v1, new_v1)
        v2_new = torch.where(diag1_len > diag2_len, new_v2, v2)
        v3_new = torch.where(diag1_len > diag2_len, v3, new_v3)
        v4_new = torch.where(diag1_len > diag2_len, new_v4, v4)

        return torch.stack((v1_new, v2_new, v3_new, v4_new), dim=-2)

    def __vertices_to_theta(self, vertices: torch.Tensor, specific: Encodings):
        # transform rectangular vertices to (x, y, w, h, theta)
        # note: theta is in rad!
        # with x,y being center coordinates of box and theta 
        # correponding to the theta as defined by the mmcv RoiAlignRotated 
        vertices = self.__parallelogram_vertices_to_rectangular_vertices(vertices)

        if specific == Encodings.THETA_FORMAT_TL_RT:
            # CLOCKWISE ANGLE
            tl_vertices = self.__get_top_left_vertices(vertices)
            bl_vertices = self.__get_bot_left_vertices(vertices)
            rt_vertices = self.__get_right_top_vertices(vertices)
            lb_vertices = self.__get_left_bot_vertices(vertices)
            lt_vertices = self.__get_left_top_vertices(vertices)
            ref_vector = rt_vertices - tl_vertices
            # reversed y-axis (clockwise angle)
            ref_vector[..., 1] = ref_vector[..., 1] * -1

            width = torch.norm(ref_vector, dim=-1)
            height = torch.norm(bl_vertices - rt_vertices, dim=-1)
            center = rt_vertices + (lb_vertices - rt_vertices) / 2

        elif specific == Encodings.THETA_FORMAT_BL_RB:
            # COUNTER CLOCKWISE ANGLE
            tl_vertices = self.__get_top_left_vertices(vertices)
            bl_vertices = self.__get_bot_left_vertices(vertices)
            rb_vertices = self.__get_right_bot_vertices(vertices)
            lt_vertices = self.__get_left_top_vertices(vertices)
            ref_vector = rb_vertices - bl_vertices

            width = torch.norm(ref_vector, dim=-1)
            height = torch.norm(lt_vertices - bl_vertices, dim=-1)
            center = lt_vertices + (rb_vertices - lt_vertices) / 2
        else:
            raise ValueError(f"{specific} not supported as theta format")

        # then, we can compute the angle between this reference vector and the x axis
        rel_x_axis = torch.stack((ref_vector[..., 0], torch.zeros_like(ref_vector[..., 1])), dim=-1).to(ref_vector.device)
        angle = torch.arccos(
            torch.clamp((ref_vector[..., 0] * rel_x_axis[..., 0] + ref_vector[..., 1] * rel_x_axis[..., 1]) / 
            torch.clamp(torch.norm(ref_vector, p=2, dim=-1) * torch.norm(rel_x_axis, p=2, dim=-1), min=EPS), min=-1+EPS, max=1-EPS)
        ) 
        if specific == Encodings.THETA_FORMAT_BL_RB:
            angle = angle * -1

        x_center = center[..., 0]
        y_center = center[..., 1]

        # axis aligned -> switch width and height
        temp_height = torch.where(angle == np.pi / 2, width, height)
        width = torch.where(angle == np.pi / 2, height, width)
        height = temp_height
        angle = angle * 180 / np.pi

        return torch.cat((x_center, y_center, width, height, angle), dim=-1)

    def __theta_to_vertices(self, vertices: torch.Tensor, specific_src: Encodings):
        # center x, center y, width, height, angle
        device = vertices.device
        x = vertices[..., 0]
        y = vertices[..., 1]
        width = vertices[..., 2]
        height = vertices[..., 3]
        angle = vertices[..., 4]
        rel_x_axis = torch.stack((x, torch.zeros_like(y)), dim=-1).to(device)
        if specific_src == Encodings.THETA_FORMAT_TL_RT:
            # center is x,y
            # reference vector is 1,y
            raise NotImplementedError("do we need this?")
        elif specific_src == Encodings.THETA_FORMAT_BL_RB:
            raise NotImplementedError("do we need this?")
        else:
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
