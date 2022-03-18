#!/bin/bash
install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_SLOG_PRINT_TO_STDOUT=1
datasets_path="/opt/npu/datasets/coco"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_dataset
python3.7 CenterNet_preprocess.py ${datasets_path}/val2017 ./prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python3.7 get_info.py bin ./prep_dataset ./prep_bin.info 512 512
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=./CenterNet_bs1.om -input_text_path=./prep_bin.info -input_width=512 -input_height=512 -output_binary=True -useDvpp=False 
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

#./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=16 -om_path=test/CenterNet_bs16.om -input_text_path=./prep_bin.info -input_width=512 -input_height=512 -output_binary=True -useDvpp=False
#if [ $? != 0 ]; then
#    echo "fail!"
#    exit -1
#fi

echo "====accuracy data bs1===="
python3.7 CenterNet_postprocess.py --bin_data_path=./result/dumpOutput_device0/ 
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

#echo "====accuracy data bs16===="
#python3.7 CenterNet_postprocess.py --bin_data_path=./result/dumpOutput_device1/
#if [ $? != 0 ]; then
#    echo "fail!"
#    exit -1
#fi

echo "====performance data bs1===="
python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

#echo "====performance data bs16===="
#python3.7 test/parse.py result/perf_vision_batchsize_16_device_1.txt
#if [ $? != 0 ]; then
#    echo "fail!"
#    exit -1
#fi
echo "success"
