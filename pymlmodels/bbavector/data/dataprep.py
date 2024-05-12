from .data_augment import PhotometricDistort
import cv2
import torch
import numpy as np
import math
from .draw_gaussian import draw_umich_gaussian, gaussian_radius
from .transforms import random_flip, load_affine_matrix, random_crop_info, ex_box_jaccard
import cv2
from ..utils import BBAVectorInput, BBAVectorAnnotation

class DataPrep:
    def __init__(self, num_classes, input_h, input_w, down_ratio, max_objs):
        super(DataPrep, self).__init__()
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.max_objs = max_objs
        self.image_distort = PhotometricDistort()

    def run(self, images: list[np.ndarray], boxes: list[np.ndarray], cls: list[np.ndarray], device: torch.device):
        _input_images = []
        _input_target_orientations = []
        _input_target_heatmaps = []
        _input_target_offsets = []
        _input_target_masks = []
        _input_target_indices = []
        _input_target_vectors = []

        for i, b, c in zip(images, boxes, cls):
            img, ann = self.data_transform(i, {"pts": b, "cat": c})
            gt = self.generate_ground_truth(img, ann)
            _input_images.append(torch.tensor(gt["input"], device=device))
            _input_target_heatmaps.append(torch.tensor(gt["hm"], device=device))
            _input_target_orientations.append(torch.tensor(gt["cls_theta"], device=device))
            _input_target_offsets.append(torch.tensor(gt["reg"], device=device))
            _input_target_masks.append(torch.tensor(gt["reg_mask"], device=device))
            _input_target_indices.append(torch.tensor(gt["ind"], device=device))
            _input_target_vectors.append(torch.tensor(gt["wh"], device=device))

        return BBAVectorInput(
            image=torch.stack(_input_images, dim=0),
            annotation=BBAVectorAnnotation(
                target_orientation=torch.stack(_input_target_orientations, dim=0),
                target_heatmap=torch.stack(_input_target_heatmaps, dim=0),
                target_offset=torch.stack(_input_target_offsets, dim=0),
                target_mask=torch.stack(_input_target_masks, dim=0),
                target_index=torch.stack(_input_target_indices, dim=0),
                target_vector=torch.stack(_input_target_vectors, dim=0)
            )
        )

    def data_transform(self, image, annotation):
        out_annotations = {}
        size_thresh = 3
        out_rects = []
        out_cat = []
        if len(annotation["pts"]) == 1:
            annotation["cat"] = annotation["cat"][None,]
        for pt_old, cat in zip(annotation['pts'] , annotation['cat']):
            rect = cv2.minAreaRect(pt_old/self.down_ratio)
            if rect[1][0]<size_thresh and rect[1][1]<size_thresh:
                continue
            out_rects.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
            out_cat.append(cat)
        out_annotations['rect'] = np.asarray(out_rects, np.float32)
        out_annotations['cat'] = np.asarray(out_cat, np.uint8)
        return image, out_annotations

    def cal_bbox_wh(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        return x2-x1, y2-y1

    def cal_bbox_pts(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        bl = [x1, y2]
        tl = [x1, y1]
        tr = [x2, y1]
        br = [x2, y2]
        return np.asarray([bl, tl, tr, br], np.float32)

    def reorder_pts(self, tt, rr, bb, ll):
        pts = np.asarray([tt,rr,bb,ll],np.float32)
        l_ind = np.argmin(pts[:,0])
        r_ind = np.argmax(pts[:,0])
        t_ind = np.argmin(pts[:,1])
        b_ind = np.argmax(pts[:,1])
        tt_new = pts[t_ind,:]
        rr_new = pts[r_ind,:]
        bb_new = pts[b_ind,:]
        ll_new = pts[l_ind,:]
        return tt_new,rr_new,bb_new,ll_new


    def generate_ground_truth(self, image, annotation):
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        image = self.image_distort(np.asarray(image, np.float32))
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        image = np.transpose(image / 255. - 0.5, (2, 0, 1))

        image_h = self.input_h // self.down_ratio
        image_w = self.input_w // self.down_ratio

        hm = np.zeros((self.num_classes, image_h, image_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 10), dtype=np.float32)
        ## add
        cls_theta = np.zeros((self.max_objs, 1), dtype=np.float32)
        ## add end
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        num_objs = min(annotation['rect'].shape[0], self.max_objs)
        for k in range(num_objs):
            rect = annotation['rect'][k, :]
            cen_x, cen_y, bbox_w, bbox_h, theta = rect
            # print(theta)
            radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
            radius = max(0, int(radius))
            ct = np.asarray([cen_x, cen_y], dtype=np.float32)

            ct_int = ct.astype(np.int32)
            if len(hm[annotation['cat'][k]].shape) > 2:
                obj_hm = hm[annotation['cat'][k]][0]
            else:
                obj_hm = hm[annotation['cat'][k]]
            hm[annotation['cat'][k]] = draw_umich_gaussian(obj_hm, ct_int, radius)
            ind[k] = min(ct_int[1], image_w - 1) * image_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            # generate wh ground_truth
            pts_4 = cv2.boxPoints(((cen_x, cen_y), (bbox_w, bbox_h), theta))  # 4 x 2

            bl = pts_4[0,:]
            tl = pts_4[1,:]
            tr = pts_4[2,:]
            br = pts_4[3,:]

            tt = (np.asarray(tl,np.float32)+np.asarray(tr,np.float32))/2
            rr = (np.asarray(tr,np.float32)+np.asarray(br,np.float32))/2
            bb = (np.asarray(bl,np.float32)+np.asarray(br,np.float32))/2
            ll = (np.asarray(tl,np.float32)+np.asarray(bl,np.float32))/2

            if theta in [-90.0, -0.0, 0.0]:  # (-90, 0]
                tt,rr,bb,ll = self.reorder_pts(tt,rr,bb,ll)
            # rotational channel
            wh[k, 0:2] = tt - ct
            wh[k, 2:4] = rr - ct
            wh[k, 4:6] = bb - ct
            wh[k, 6:8] = ll - ct
            w_hbbox, h_hbbox = self.cal_bbox_wh(pts_4)
            wh[k, 8:10] = 1. * w_hbbox, 1. * h_hbbox
            jaccard_score = ex_box_jaccard(pts_4.copy(), self.cal_bbox_pts(pts_4).copy())
            if jaccard_score<0.95:
                cls_theta[k, 0] = 1

        ret = {'input': image,
               'hm': hm,
               'reg_mask': reg_mask,
               'ind': ind,
               'wh': wh,
               'reg': reg,
               'cls_theta':cls_theta,
               }
        return ret

