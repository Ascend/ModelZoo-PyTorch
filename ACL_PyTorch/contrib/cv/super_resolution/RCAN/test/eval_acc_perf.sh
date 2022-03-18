#!/bin/bash
datasets_path="/root/datasets"
for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done
arch=`uname -m`

# prepare data
if [ -d "prep_data" ]
then 
    echo "prep_data exists, auto-delete."
    rm -r prep_data
fi
python3.7 rcan_preprocess.py -s ${datasets_path}/Set5/LR -d ./prep_data --size 256
if [ $? != 0 ]; then
    echo "fail to preprocess data!"
    exit -1
fi

# get .info file
if [ -f "rcan_prep_bin.info" ]
then
    echo "info file exists, auto-delete."
    rm rcan_prep_bin.info
fi
python3.7 gen_dataset_info.py bin ./prep_data ./rcan_prep_bin.info 256 256
if [ $? != 0 ]; then
    echo "fail to get info file!"
    exit -1
fi

# config environment
source env.sh
# run om
if [ -d "result" ]
then
    echo "om running result exists, auto-delete."
    rm -r result
fi
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=rcan_1bs.om -input_text_path=./rcan_prep_bin.info -input_width=256 -input_height=256 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail to run 1bs om!"
    exit -1
fi

# postprocess data
if [ -d "post_data" ]
then
    echo "post data exists, auto-delete."
    rm -r post_data
fi 
python3.7 rcan_postprocess.py -s result/dumpOutput_device0/ -d post_data
if [ $? != 0 ]; then
    echo "fail to postprocess data!"
    exit -1
fi

# evaluate the result
python3.7 evaluate.py --infer post_data --HR ${datasets_path}/Set5/HR --result result_1bs.json
if [ $? != 0 ]; then
    echo "fail to calculate accurancy of 1bs om!"
    exit -1
fi

# summary
echo "====accuracy data===="
python3.7 test/parse.py result_1bs.json
if [ $? != 0 ]; then
    echo "fail to show accurancy data of 1bs!"
    exit -1
fi
echo "====performance data===="
python test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail to show performance data of 1bs!"
    exit -1
fi

echo "success"
