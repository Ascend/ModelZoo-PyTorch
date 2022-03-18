#!/bin/bash

datasets_path="/root/datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./val2017_ssd_bin
python3.7 mmdetection_coco_preprocess.py --image_folder_path ${datasets_path}/coco/val2017 --bin_folder_path val2017_ssd_bin
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "preprocess"

python3.7 get_info.py bin ./val2017_ssd_bin coco2017_ssd.info 300 300
python3.7 get_info.py jpg ${datasets_path}/coco/val2017 coco2017_ssd_jpg.info
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "get_info"

source env.sh
rm -rf result/dumpOutput_device0
chmod u+x benchmark.x86_64
./benchmark.x86_64 -model_type=vision -batch_size=1 -device_id=0 -input_text_path=./coco2017_ssd.info -input_width=300 -input_height=300 -useDvpp=False -output_binary=true -om_path=ssd_300_coco.om
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "benchmark.x86_64"

python3.7 mmdetection_coco_postprocess.py --bin_data_path=result/dumpOutput_device0 --prob_thres=0.02 --ifShowDetObj --det_results_path=detection-results --test_annotation=coco2017_ssd_jpg.info
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "postprocess"

python3.7 txt_to_json.py
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "txt_to_json"

python3.7 coco_eval.py  --ground_truth ${datasets_path}/coco/annotations/instances_val2017.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "coco_eval"

echo "====performance data===="
python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"
