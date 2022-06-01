#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
datasets_path="/opt/npu/coco" #该路径根据实际数据集地址更改，但必须的是需在主路径下的./CenterNet/data/coco/中放入数据集，便于后续推理执行

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
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=./CenterNet_bs1_710.om -input_text_path=./prep_bin.info -input_width=512 -input_height=512 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

#./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=32 -om_path=test/CenterNet_bs32_710.om -input_text_path=./prep_bin.info -input_width=512 -input_height=512 -output_binary=True -useDvpp=False
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

#echo "====accuracy data bs32===="
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

#echo "====performance data bs32===="
#python3.7 test/parse.py result/perf_vision_batchsize_32_device_1.txt
#if [ $? != 0 ]; then
#    echo "fail!"
#    exit -1
#fi
echo "success"