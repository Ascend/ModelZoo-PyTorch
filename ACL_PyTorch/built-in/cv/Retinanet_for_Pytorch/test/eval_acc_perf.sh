#!/bin/bash

datasets_path="/root/datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

rm -rf ./val2017_bin
python3.7 retinanet_pth_preprocess_detectron2.py --image_src_path=${datasets_path}/coco/val2017 --bin_file_path=val2017_bin --model_input_height=1344 --model_input_width=1344
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 get_info.py bin ./val2017_bin retinanet.info 1344 1344
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 get_info.py jpg ${datasets_path}/coco/val2017 origin_image.info
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source env.sh
rm -rf result/dumpOutput_device0
./benchmark.x86_64 -model_type=vision -om_path=retinanet_detectron2_npu.om -device_id=0 -batch_size=1 -input_text_path=retinanet.info -input_width=1344 -input_height=1344 -useDvpp=false -output_binary=true
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 retinanet_pth_postprocess_detectron2.py --bin_data_path=./result/dumpOutput_device0/ --val2017_path=${datasets_path}/coco --test_annotation=origin_image.info --det_results_path=./ret_npuinfer/ --net_out_num=3 --net_input_height=1344 --net_input_width=1344 
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