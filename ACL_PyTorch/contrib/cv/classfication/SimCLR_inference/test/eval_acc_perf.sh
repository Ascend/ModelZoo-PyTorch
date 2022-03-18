!/bin/bash

datasets_path="/root/datasets"
for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_data
python3.7 Simclr_preprocess.py $datasets_path/cifar-10-batches-py/test_batch ./prep_data
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 gen_dataset_info.py bin ./prep_data ./Simclr_model.info 32 32
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source env.sh
rm -rf result/dumpOutput_device0
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./Simclr_model_bs1.om -input_text_path=./Simclr_model.info -input_width=32 -input_height=32 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python3.7 Simclr_postprocess.py  ./result/dumpOutput_device0/ > result_bs1.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=16 -om_path=./Simclr_model_bs16.om -input_text_path=./Simclr_model.info -input_width=32 -input_height=32 -output_binary=False -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python3.7 Simclr_postprocess.py  ./result/dumpOutput_device0/ > result_bs16.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo -e "\n"
echo "==== accuracy data ===="

echo "==== 310 bs1 PSNR ===="
python3.7 test/parse.py result_bs1.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "==== 310 bs16 PSNR ===="
python3.7 test/parse.py result_bs16.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo -e "\n"
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
