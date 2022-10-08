#!/bin/bash

datasets_path="./data/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

echo "===========  start preprocess  =============="
echo "img shape = 3*1024*1024"
rm -rf ./preprocess_bin
python3.7 Wseg_preprocess.py ${datasets_path} ./val_voc.txt ./preprocess_bin
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "get_dateset_info start!"
python3.7 get_dateset_info.py bin ./preprocess_bin ./prep_bin.info 1024 1024
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo "===========  start benchmark  =============="
echo "benchmark batch1"
rm -rf result/dumpOutput_device1
./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=1 -om_path=./wideresnet_bs1.om -input_text_path=./prep_bin.info -input_width=1024 -input_height=1024 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "benchmark batch4"
rm -rf result/dumpOutput_device2
./benchmark.x86_64 -model_type=vision -device_id=2 -batch_size=4 -om_path=./wideresnet_bs4.om -input_text_path=./prep_bin.info -input_width=1024 -input_height=1024 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "===========  start postprocess  =============="
echo 'postprocess batch_size = 1'
python3.6 Wseg_postprocess.py ${datasets_path} ./val_voc.txt ./result/dumpOutput_device1/ ./output/output_batch1/
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo 'postprocess batch_size = 4'
python3.6 Wseg_postprocess.py ${datasets_path} ./val_voc.txt ./result/dumpOutput_device2/ ./output/output_batch4/
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "===========  start eval  =============="
echo "eval batch1 "
python3.7 ./wseg/eval_seg.py --data ${datasets_path} --filelist './val_voc.txt' --masks 'output/output_batch1/crf' > 'output/output_batch1/crf.eval' 2>&1 &
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "eval batch4"
python3.7 ./wseg/eval_seg.py --data ${datasets_path} --filelist './val_voc.txt' --masks 'output/output_batch4/crf' > 'output/output_batch4/crf.eval' 2>&1 &
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
sleep 5
echo "acc result  baseline IOU:59.7"
echo "batch 1"
python3.7 ./test/parse.py ./output/output_batch1/crf.eval
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "batch 4"
python3.7 ./test/parse.py ./output/output_batch4/crf.eval
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "performance result"
python3.7 test/parse.py result/perf_vision_batchsize_1_device_1.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py result/perf_vision_batchsize_4_device_2.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"