Extracted and slightly modified from https://github.com/facebookresearch/detectron2

Copyright (c) Facebook, Inc. and its affiliates.

Changes:

- removed unecessary checks and bindings from vision.cpp
- removed all "detectron2" namespaces from all .cpp, .cu, .h files
- also used the detectron2 pytorch code for orientedrcnn/nms_rotated.py and orientedrcnn/oriented_rcnn_head_roi_align_rotated.py
