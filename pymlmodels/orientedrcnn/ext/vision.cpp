// Copyright (c) Facebook, Inc. and its affiliates.

#include <torch/extension.h>
#include "ROIAlignRotated/ROIAlignRotated.h"
#include "nms_rotated/nms_rotated.h"
#include "box_iou_rotated/box_iou_rotated.h"
#include "box_iou_rotated_mmcv/box_iou_rotated_mmcv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms_rotated", &nms_rotated);
  m.def("box_iou_rotated", &box_iou_rotated);
  m.def("box_iou_rotated_mmcv", &box_iou_rotated_mmcv);
  m.def("roi_align_rotated_forward", &ROIAlignRotated_forward);
  m.def("roi_align_rotated_backward", &ROIAlignRotated_backward);
}
