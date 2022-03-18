#!/bin/bash
datasets_path="/root/datasets"
for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done
arch=`uname -m`
rm -rf ./val2017_bin
python3.7 gfocal_preprocess.py --image_src_path=${datasets_path}/coco/val2017 --bin_file_path=val2017_bin --model_input_height=800 --model_input_width=1216
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 get_info.py bin val2017_bin gfocal.info 1216 800
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 get_info.py jpg ${datasets_path}/coco/val2017 gfocal_jpeg.info
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -om_path=gfocal_bs1.om -device_id=0 -batch_size=1 -input_text_path=gfocal.info -input_width=1216 -input_height=800 -useDvpp=false -output_binary=true
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 gfocal_postprocess.py --bin_data_path=./result/dumpOutput_device0/ --test_annotation=gfocal_jpeg.info --net_out_num=3 --net_input_height=800 --net_input_width=1216
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