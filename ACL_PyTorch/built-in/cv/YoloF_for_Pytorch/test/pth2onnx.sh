#!/bin/bash

set -eu
batch_size=16
mmdetection_path=$1
mmdeploy_path=$2
cp yolof_r50_c5_8x8_1x_coco.py ${mmdetection_path}/configs/yolof
cp bbox_nms.py ${mmdeploy_path}/mmdeploy/codebase/mmdet/core/post_processing/bbox_nms.py
cd ${mmdeploy_path}
python tools/deploy.py \
configs/mmdet/detection/detection_onnxruntime_dynamic.py \
${mmdetection_path}/configs/yolof/yolof_r50_c5_8x8_1x_coco.py  \
yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth \
${mmdetection_path}/demo/demo.jpg \
--work-dir work_dir
