  #!/bin/bash

datasets_path=/opt/npu/
arch=`uname -m`
echo $arch

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

echo "====数据预处理===="
rm -rf ./pre_dataset_bin
python3.7 pre_dataset.py ${datasets_path}/cityscapes ./pre_dataset_bin
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====数据预处理完成===="


echo "====生成数据info文件===="
rm -rf ./icnet_pre_bin_1024_2048.info
python3.7 get_info.py bin ./pre_dataset_bin ./icnet_pre_bin_1024_2048.info 1024 2048
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====生成数据info文件完成===="


echo "====bs1 benchmark推理===="
source env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=./ICNet_bs1.om -input_text_path=./icnet_pre_bin_1024_2048.info -input_width=1024 -input_height=2048 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====bs1 benchmark推理完成===="


echo "====bs4 benchmark推理===="
source env.sh
rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=4 -om_path=./ICNet_bs4.om -input_text_path=./icnet_pre_bin_1024_2048.info -input_width=1024 -input_height=2048 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====bs4 benchmark推理完成===="


echo "====bs1 精度后处理===="
rm -rf icnet_bs1.log
python3.7 -u evaluate.py ${datasets_path}/cityscapes ./result/dumpOutput_device0 ./out >icnet_bs1.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====bs1 精度后处理完成===="


echo "====bs4 精度后处理===="
rm -rf icnet_bs4.log
python3.7 -u evaluate.py ${datasets_path}/cityscapes ./result/dumpOutput_device1 ./out >icnet_bs4.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====bs4 精度后处理完成===="

echo "====accuracy data===="
python3.7 test/parse.py icnet_bs1.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python3.7 test/parse.py icnet_bs4.log
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


python3.7 test/parse.py result/perf_vision_batchsize_4_device_1.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"