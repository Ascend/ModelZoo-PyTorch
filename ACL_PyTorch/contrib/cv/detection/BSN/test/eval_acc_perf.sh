#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo "====TEM pre_treatment starting===="
python BSN_tem_preprocess.py 
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====TEM pre_treatment finished===="

echo "====creating TEM info file===="

python gen_dataset_info.py tem ./output/BSN-TEM-preprocess/feature TEM-video-feature 400 100
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====creating TEM info file done===="


echo "====TEM bs1 benchmark start===="

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=BSN_tem_bs1.om -input_text_path=./TEM-video-feature.info -input_width=400 -input_height=100 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====TEM bs1 benchmark finished===="

echo "====TEM bs1 postprocess===="

python BSN_tem_postprocess.py
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====TEM bs1 posrprocess finished===="

echo "====PEM pre_treatment starting===="
python BSN_pem_preprocess.py 
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====PEM pre_treatment finished===="

echo "====creating PEM info file===="

python gen_dataset_info.py pem output/BSN-PEM-preprocess/feature PEM-video-feature 1000 32
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====creating PEM info file done===="

echo "====PEM bs1 benchmark start===="

./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=1 -om_path=BSN_pem_bs1.om -input_text_path=./PEM-video-feature.info -input_width=1000 -input_height=32 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====PEM bs1 benchmark finished===="

echo "====PEM bs1 postprocess===="

python BSN_pem_postprocess.py
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====PEM bs1 posrprocess finished===="

echo "====bs1 evaluate start===="

python BSN_eval.py >BSN_bs1.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====bs1 evaluate finished===="

echo "====TEM bs16 benchmark start===="
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=BSN_tem_bs16.om -input_text_path=./TEM-video-feature.info -input_width=400 -input_height=100 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====TEM bs16 benchmark finished===="

echo "====TEM bs16 postprocess===="

python BSN_tem_postprocess.py
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====TEM bs16 posrprocess finished===="

echo "====PEM pre_treatment starting===="
python BSN_pem_preprocess.py 
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====PEM pre_treatment finished===="

echo "====creating PEM info file===="

python gen_dataset_info.py pem output/BSN-PEM-preprocess/feature PEM-video-feature 1000 32
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====creating PEM info file done===="

echo "====PEM bs16 benchmark start===="

./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=16 -om_path=BSN_pem_bs16.om -input_text_path=./PEM-video-feature.info -input_width=1000 -input_height=32 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====PEM bs16 benchmark finished===="

echo "====PEM bs16 postprocess===="

python BSN_pem_postprocess.py
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====PEM bs16 posrprocess finished===="

echo "====bs16 evaluate start===="

python BSN_eval.py >BSN_bs16.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====bs16 evaluate finished===="

echo "====accuracy data bs1===="

python parse.py BSN_bs1.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data TEM bs1===="
python parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data PEM bs1===="
python parse.py result/perf_vision_batchsize_1_device_1.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data bs16===="

python parse.py BSN_bs16.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data TEM bs16===="
python parse.py result/perf_vision_batchsize_16_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data PEM bs16===="
python parse.py result/perf_vision_batchsize_16_device_1.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"