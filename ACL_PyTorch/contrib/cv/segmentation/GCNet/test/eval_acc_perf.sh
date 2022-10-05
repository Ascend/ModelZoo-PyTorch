#!/bin/bash
export datasets_path=`pwd`/mmdetection/data/coco/
for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done
arch=`uname -m`
rm -rf ./val2017_bin
rm -rf ./coco2017.info
rm -rf ./coco2017_jpg.info
echo "====preprocess===="
python GCNet_preprocess.py --image_src_path=${datasets_path}/val2017 --bin_file_path=val2017_bin --model_input_height=800 --model_input_width=1216
if [ $? != 0 ]; then
    echo "fail!"				        
    exit -1
fi
echo "====get_info===="
python gen_dataset_info.py bin val2017_bin coco2017.info 1216 800
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python gen_dataset_info.py jpg ${datasets_path}/val2017 coco2017_jpg.info
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result/dumpOutput_device1
./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=1 -om_path=./GCNet_bs1.om -input_text_path=./coco2017.info  -input_width=1216 -input_height=800 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python GCNet_postprocess.py --bin_data_path=./result/dumpOutput_device1/ --test_annotation=coco2017_jpg.info --det_results_path=detection-results --annotations_path=${datasets_path}/annotations/instances_val2017.json --net_out_num=3 --net_input_height=800 --net_input_width=1216
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
python txt_to_json.py
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python coco_eval.py --ground_truth=${datasets_path}/annotations/instances_val2017.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data===="
./benchmark.x86_64 -round=20 -om_path=GCNet_bs1.om -device_id=1 -batch_size=1
echo "success"
