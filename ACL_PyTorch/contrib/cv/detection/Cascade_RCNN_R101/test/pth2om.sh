#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

rm -rf cascade_rcnn_r101.onnx
python mmdetection/tools/pytorch2onnx.py mmdetection/configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py ./cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth --output-file=cascade_rcnn_r101.onnx --shape 1216 --verify --show
source env.sh
rm -rf cascade_rcnn_r101_1.om
atc --framework=5 --model=./cascade_rcnn_r101.onnx --output=cascade_rcnn_r101_1 --input_format=NCHW --input_shape="input:1,3,1216,1216" --soc_version=Ascend310 --out_nodes="Concat_947:0;Reshape_949:0"

if [ -f "cascade_rcnn_r101_1.om" ] ; then
    echo "success"
else
    echo "fail!"
fi