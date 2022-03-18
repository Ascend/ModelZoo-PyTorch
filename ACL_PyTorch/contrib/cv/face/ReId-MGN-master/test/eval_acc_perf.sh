#!/bin/bash

datasets_path="./data/Market1501/"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done
# batch sizeΪ1
rm -rf ./data/Market1501/bin_data/q ./data/Market1501/bin_data/g ./data/Market1501/bin_data_flip/q ./data/Market1501/bin_data_flip/g
mkdir ./data/Market1501/bin_data/q -p
mkdir ./data/Market1501/bin_data/g -p
mkdir ./data/Market1501/bin_data_flip/q -p
mkdir ./data/Market1501/bin_data_flip/g -p
python3.7 ./postprocess_MGN.py  --mode save_bin --data_path ./data/Market1501/
if [ $? != 0 ]; then
    echo "create bin data fail!"
    exit -1
fi
python3.7 ./preprocess_MGN.py bin ./data/Market1501/bin_data/q/ ./q_bin.info 384 128
if [ $? != 0 ]; then
    echo "create q bin data info fail!"
    exit -1
fi
python3.7 ./preprocess_MGN.py bin ./data/Market1501/bin_data/g/ ./g_bin.info 384 128
if [ $? != 0 ]; then
    echo "create g bin data info fail!"
    exit -1
fi
python3.7 ./preprocess_MGN.py bin ./data/Market1501/bin_data_flip/q/ ./q_bin_flip.info 384 128
if [ $? != 0 ]; then
    echo "create flipped q bin data info fail!"
    exit -1
fi
python3.7 ./preprocess_MGN.py bin ./data/Market1501/bin_data_flip/g/ ./g_bin_flip.info 384 128
if [ $? != 0 ]; then
    echo "create flipped g bin data info fail!"
    exit -1
fi
source env.sh
mkdir ./result -p
rm -rf ./result/dumpOutput_device0 ./result/q_bin ./result/g_bin ./result/q_bin_flip ./result/g_bin_flip
mkdir ./result/q_bin -p
mkdir ./result/g_bin -p
mkdir ./result/q_bin_flip -p
mkdir ./result/g_bin_flip -p
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=mgn_mkt1501_bs1.om -input_text_path=./q_bin.info -input_width=384 -input_height=128 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "get q output of om fail!"
    exit -1
fi
mv ./result/dumpOutput_device0 ./result/q_bin
rm -rf ./result/dumpOutput_device0
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=mgn_mkt1501_bs1.om -input_text_path=./g_bin.info -input_width=384 -input_height=128 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "get g output of om fail!!"
    exit -1
fi
mv ./result/dumpOutput_device0 ./result/g_bin
rm -rf ./result/dumpOutput_device0
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=mgn_mkt1501_bs1.om -input_text_path=./q_bin_flip.info -input_width=384 -input_height=128 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "get flipped q output of om fail!"
    exit -1
fi
mv ./result/dumpOutput_device0 ./result/q_bin_flip
rm -rf ./result/dumpOutput_device0
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=mgn_mkt1501_bs1.om -input_text_path=./g_bin_flip.info -input_width=384 -input_height=128 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "get flipped g output of om fail!"
    exit -1
fi
mv ./result/dumpOutput_device0 ./result/g_bin_flip
rm -rf ./result/dumpOutput_device0
python3.7 ./postprocess_MGN.py  --mode evaluate_om --data_path ./data/Market1501/
if [ $? != 0 ]; then
    echo "get acc fail!"
    exit -1
else
    echo "batch size 1 success!"
fi
# batch sizeΪ16
mkdir ./result -p
rm -rf ./result/dumpOutput_device0
rm -rf ./result/dumpOutput_device0 ./result/q_bin ./result/g_bin ./result/q_bin_flip ./result/g_bin_flip
mkdir ./result/q_bin -p
mkdir ./result/g_bin -p
mkdir ./result/q_bin_flip -p
mkdir ./result/g_bin_flip -p
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=mgn_mkt1501_bs16.om -input_text_path=./q_bin.info -input_width=384 -input_height=128 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "get q output of om fail!"
    exit -1
fi
mv ./result/dumpOutput_device0 ./result/q_bin
rm -rf ./result/dumpOutput_device0
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=mgn_mkt1501_bs16.om -input_text_path=./g_bin.info -input_width=384 -input_height=128 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "get g output of om fail!!"
    exit -1
fi
mv ./result/dumpOutput_device0 ./result/g_bin
rm -rf ./result/dumpOutput_device0
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=mgn_mkt1501_bs16.om -input_text_path=./q_bin_flip.info -input_width=384 -input_height=128 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "get flipped q output of om fail!"
    exit -1
fi
mv ./result/dumpOutput_device0 ./result/q_bin_flip
rm -rf ./result/dumpOutput_device0
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=mgn_mkt1501_bs16.om -input_text_path=./g_bin_flip.info -input_width=384 -input_height=128 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "get flipped g output of om fail!"
    exit -1
fi
mv ./result/dumpOutput_device0 ./result/g_bin_flip
rm -rf ./result/dumpOutput_device0
python3.7 ./postprocess_MGN.py  --mode evaluate_om --data_path ./data/Market1501/
if [ $? != 0 ]; then
    echo "get acc fail!"
    exit -1
else
    echo "batch size 16 success!"
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