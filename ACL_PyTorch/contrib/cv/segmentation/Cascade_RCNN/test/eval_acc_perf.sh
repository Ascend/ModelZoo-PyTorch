#!/bin/bash

datasets_path="/root/datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_dataset
python3.7 cascadercnn_pth_preprocess_detectron2.py --image_src_path=${datasets_path}/coco/val2017 --bin_file_path=val2017_bin --model_input_height=1344 --model_input_width=1344
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 get_info.py bin val2017_bin cascadercnn.info 1344 1344
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -om_path=cascadercnn_detectron2_npu.om -device_id=0 -batch_size=1 -input_text_path=cascadercnn.info -input_width=1344 -input_height=1344 -useDvpp=false -output_binary=true
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 get_info.py jpg ${datasets_path}/coco/val2017 cascadercnn_jpeg.info
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
echo "pkl bbox AP:43.9%"
python3.7 cascadercnn_pth_postprocess_detectron2.py --bin_data_path=./result/dumpOutput_device0/ --test_annotation=cascadercnn_jpeg.info --det_results_path=./ret_npuinfer/ --net_out_num=4 --net_input_height=1344 --net_input_width=1344 --ifShowDetObj
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data===="
python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"