#!/bin/bash

datasets_path="/root/datasets/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf prep_dataset
python3.7 AdvancedEAST_preprocess.py ${datasets_path}/icpr prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf prep_bin.info
python3.7 gen_dataset_info.py bin prep_dataset prep_bin.info 736 736
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -batch_size=1 -device_id=0 -om_path=AdvancedEAST_bs1.om -input_text_path=prep_bin.info -input_width=736 -input_height=736 -useDvpp=false -output_binary=true
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf eval_temp result_bs1.log
python3.7 AdvancedEAST_postprocess.py ${datasets_path}/icpr result/dumpOutput_device0 > result_bs1.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -batch_size=16 -device_id=0 -om_path=AdvancedEAST_bs16.om -input_text_path=prep_bin.info -input_width=736 -input_height=736 -useDvpp=false -output_binary=true
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf eval_temp result_bs16.log
python3.7 AdvancedEAST_postprocess.py ${datasets_path}/icpr result/dumpOutput_device0 > result_bs16.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
python3.7 test/parse.py result_bs1.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py result_bs16.log
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
python3.7 test/parse.py result/perf_vision_batchsize_16_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"
