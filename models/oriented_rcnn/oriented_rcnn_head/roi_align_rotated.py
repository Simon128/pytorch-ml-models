from mmcv.ops import RoIAlignRotated, nms
import torch
from collections import OrderedDict

from ..data_formats import RPNOutput
from ..encoder import encode, Encodings

class RoIAlignRotatedWrapper(RoIAlignRotated):
    def __init__(self, output_size, spatial_scale, sampling_ratio: int, fpn_strides = [4, 8, 16, 32, 64]):
        self.fpn_level_scalings = fpn_strides
        super().__init__(output_size, spatial_scale, sampling_ratio=sampling_ratio, clockwise=True)
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
        for test_idx, (k, v) in enumerate(result.items()):
            b = rpn_proposals[k].objectness_scores.shape[0]
            cv2_format = encode(v, Encodings.VERTICES, Encodings.ORIENTED_CV2_FORMAT)
            hbb = encode(v, Encodings.VERTICES, Encodings.HBB_CORNERS)
            batch_indexed = []
            level_scores = []
            if self.test > 100:
                # debug
                import cv2
                import numpy as np
                image = cv2.imread("/home/simon/dev/pytorch-ml-models/models/oriented_rcnn/example/image.tif")
                regression = v[0]
                objectness = rpn_proposals[k].objectness_scores[0] 
                mask = torch.sigmoid(objectness) > 0.8
                if mask.count_nonzero().item() > 0:
                    top_kk = min(mask.count_nonzero().item(), 100)
                    thr_objectness = objectness[mask]
                    thr_regression = regression[mask] * self.fpn_level_scalings[test_idx]
                    top_idx = torch.topk(thr_objectness, k=int(top_kk)).indices
                    top_boxes = torch.gather(thr_regression, 0, top_idx.unsqueeze(-1).unsqueeze(-1).repeat((1, 4, 2)))

                    #for box in boxes:
                    #    rot = ((box[0].item(), box[1].item()), (box[2].item(), box[3].item()), box[4].item() * 180 / math.pi)
                    #    pts = cv2.boxPoints(rot) # type: ignore
                    #    pts = np.int0(pts) # type: ignore
                    #    image = cv2.drawContours(image, [pts], 0, (0, 255, 0), 2) # type: ignore

                    #hbb_boxes = v[b_idx, keep].squeeze(0)[:10].detach().clone().cpu().numpy().astype(np.int32)

                    thickness = 1
                    isClosed = True
                    pred_color = (0, 0, 255)

                    for o in top_boxes:
                        pts_pred = o.cpu().detach().numpy().astype(np.int32)
                        image = cv2.polylines(image, [pts_pred], isClosed, pred_color, thickness) # type: ignore

                    cv2.imwrite(f"test_{k}.png", image) # type: ignore

            for b_idx in range(b):
                # take the top 2000 rpn proposals and apply nms
                topk_k = min(2000, rpn_proposals[k].objectness_scores.shape[1])
                topk_proposals = torch.topk(rpn_proposals[k].objectness_scores[b_idx], k=topk_k)
                topk_idx = topk_proposals.indices
                topk_scores = topk_proposals.values
                nms_result = nms(hbb[b_idx, topk_idx], topk_scores, 0.5)
                keep = nms_result[1]
                kept_boxes = cv2_format[b_idx, keep]
                kept_scores = topk_scores[keep]
                n = kept_boxes.shape[0]
                b_idx_tensor = torch.full((n, 1), b_idx).to(v.device)
                values = torch.concatenate((b_idx_tensor, kept_boxes), dim=-1)
                batch_indexed.append(values)
                level_scores.append(kept_scores)

            result[k] = {}
            result[k]["boxes"] = torch.concatenate(batch_indexed, dim=0)
            result[k]["scores"] = torch.stack(level_scores, dim=0)

        return result

    def forward(self, fpn_features: OrderedDict, rpn_proposals: OrderedDict, anchors: OrderedDict):
        # this is doing the roi align rotated + the filtering described in the section 3.3 of the paper
        cv2_format = self.process_rpn_proposals(rpn_proposals, anchors)
        merged_features = []
        merged_boxes = []
        merged_scores = []

        for s_idx, k in enumerate(cv2_format.keys()):
            num_batches = fpn_features[k].shape[0]
            roi_align = super().forward(fpn_features[k], cv2_format[k]["boxes"])
            # todo: find a better way
            batched_roi_align = {b: [] for b in range(num_batches)}
            batched_boxes = {b: [] for b in range(num_batches)}
            for idx, batch_idx in enumerate(cv2_format[k]["boxes"][:, 0]):
                batched_roi_align[int(batch_idx.item())].append(roi_align[idx])
                batched_boxes[int(batch_idx.item())].append(cv2_format[k]["boxes"][idx, 1:])
            merged_features.append(torch.stack([torch.stack(bra, dim=0) for bra in batched_roi_align.values()], dim=0))
            boxes = torch.stack([torch.stack(bb, dim=0) for bb in batched_boxes.values()], dim=0)
            boxes[..., :-1] = boxes[..., :-1] * self.fpn_level_scalings[s_idx]
            merged_boxes.append(boxes)
            merged_scores.append(cv2_format[k]["scores"])

        merged_features = torch.concatenate(merged_features, dim=1)
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

