#!/bin/bash

set -eu
batch_size=64
mmdetection_path=$1
mmdeploy_path=$2
cp yolox_tiny_8x8_300e_coco.py ${mmdetection_path}/configs/yolox
cp bbox_nms.py ${mmdeploy_path}/mmdeploy/codebase/mmdet/core/post_processing/bbox_nms.py
cd ${mmdeploy_path}
python tools/deploy.py \
configs/mmdet/detection/detection_onnxruntime_dynamic.py \
${mmdetection_path}/configs/yolox/yolox_tiny_8x8_300e_coco.py  \
yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth \
${mmdetection_path}/demo/demo.jpg \
--work-dir work_dir
