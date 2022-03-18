#!/bin/bash

datasets_path=/opt/npu/FDDB/
arch=`uname -m`
echo $arch

echo "====数据预处理===="
python3.7 faceboxes_pth_preprocess.py --dataset ${datasets_path} --save-folder prep/
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====数据预处理完成===="

echo "====生成数据info文件===="
python3.7 get_info.py bin ./prep ./faceboxes_prep_bin.info 1024 1024
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====生成数据info文件完成===="

cd benchmark_tools/
rm -rf prep
cp -a ../prep ./

echo "====bs1 benchmark推理===="
source env.sh
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=../faceboxes-b0_bs1.om -input_text_path=../faceboxes_prep_bin.info -input_width=1024 -input_height=1024 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====bs1 benchmark推理完成===="

echo "====bs1 精度后处理===="
cd ..
python3.7 faceboxes_pth_postprocess.py --save_folder FDDB_Evaluation/ --prep_info prep/ --prep_folder benchmark_tools/result/dumpOutput_device0/
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
cd FDDB_Evaluation
python3.7 convert.py
python3.7 split.py
python3.7 evaluate.py -p pred_sample
echo "====bs1 精度后处理完成===="


echo "====bs16 benchmark推理===="
cd ../benchmark_tools/
source env.sh
./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=1 -om_path=../faceboxes-b0_bs1.om -input_text_path=../faceboxes_prep_bin.info -input_width=1024 -input_height=1024 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====bs16 benchmark推理完成===="

echo "====bs16 精度后处理===="
cd ..
python3.7 faceboxes_pth_postprocess.py --save_folder FDDB_Evaluation/ --prep_info prep/ --prep_folder benchmark_tools/result/dumpOutput_device1/
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
cd FDDB_Evaluation
python3.7 convert.py
python3.7 split.py
python3.7 evaluate.py -p pred_sample
echo "====bs16 精度后处理完成===="