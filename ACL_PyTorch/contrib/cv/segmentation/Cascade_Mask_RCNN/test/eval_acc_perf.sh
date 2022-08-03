#!/bin/bash

datasets_path="/opt/npu"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done
arch=`uname -m`
python3.7 cascade_maskrcnn_preprocess.py --image_src_path=${datasets_path}/coco/val2017 --bin_file_path=val2017_bin --model_input_height=1344 --model_input_width=1344
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 get_info.py bin val2017_bin cascade_maskrcnn.info 1344 1344
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 get_info.py jpg ${datasets_path}/coco/val2017 cascade_maskrcnn_jpeg.info
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./benchmark.x86_64 -model_type=vision -om_path=output/cascade_maskrcnn_bs1.om -device_id=0 -batch_size=1 -input_text_path=cascade_maskrcnn.info -input_width=1344 -input_height=1344 -useDvpp=false -output_binary=true
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 cascade_maskrcnn_postprocess.py --bin_data_path=./result/dumpOutput_device0/ --test_annotation=cascade_maskrcnn_jpeg.info --det_results_path=./ret_npuinfer/ --net_out_num=4 --net_input_height=1344 --net_input_width=1344 --ifShowDetObj
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
./benchmark.x86_64 -round=20 -om_path=output/cascade_maskrcnn_bs1.om -device_id=0 -batch_size=1
