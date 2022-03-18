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

rm -rf nas_fpn.onnx
python mmdetection/tools/pytorch2onnx.py mmdetection/configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py ./retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth --output-file=nas_fpn.onnx --shape=640 --verify --show
source env.sh
rm -rf nas_fpn_1.om
atc --framework=5 --model=./nas_fpn.onnx --output=nas_fpn_1 --input_format=NCHW --input_shape="input:1,3,640,640" --soc_version=Ascend310 --out_nodes="Concat_1487:0;Reshape_1489:0"

if [ -f "nas_fpn_1.om" ] ; then
    echo "success"
else
    echo "fail!"
fi