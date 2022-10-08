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
python3.7 get_info.py bin ${datasets_path}/coco/val2017_bin gfocal.info 1216 800
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 get_info.py jpg ${datasets_path}/coco/val2017 gfocal_jpeg.info
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf ./out/
./tools/msame/out/msame --model "./gfocal_bs1.om" --input "./val2017_bin" --output "./out/" --outfmt TXT
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python3.7 gfocal_postprocess.py --bin_data_path=./out/2* --test_annotation=gfocal_jpeg.info --net_out_num=3 --net_input_height=800 --net_input_width=1216
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