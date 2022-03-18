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

datasets_path="/opt/npu/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./val2017_bin
python mmdetection_coco_preprocess.py --image_folder_path ${datasets_path}coco/val2017 --bin_folder_path val2017_bin
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python get_info.py bin ./val2017_bin coco2017.info 1216 1216
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python get_info.py jpg ${datasets_path}coco/val2017 coco2017_jpg.info
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
chmod u+x ben*
source env.sh
rm -rf result/dumpOutput_device2
./benchmark.${arch}  -model_type=vision -batch_size=1 -device_id=2 -input_text_path=./coco2017.info -input_width=1216 -input_height=1216 -useDvpp=False -output_binary=true -om_path=cascade_rcnn_r101_1.om
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
python mmdetection_coco_postprocess.py --bin_data_path=result/dumpOutput_device2 --prob_thres=0.05 --ifShowDetObj --det_results_path=detection-results --test_annotation=coco2017_jpg.info --img_path ${datasets_path}coco/val2017
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data===="
python test/parse.py result/perf_vision_batchsize_1_device_2.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
