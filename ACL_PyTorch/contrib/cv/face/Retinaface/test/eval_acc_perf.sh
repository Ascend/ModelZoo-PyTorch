#!/bin/bash
source test/env.sh

arch=`uname -m`
currentDir=$(cd "$(dirname "$0")";pwd)/..

rm -rf ./widerface
python3.7 retinaface_pth_preprocess.py
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python3.7 get_info.py bin ./widerface/prep ./retinaface_pre_bin.info 1000 1000
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=retinaface_bs1.om -input_text_path=./retinaface_pre_bin.info -input_width=1000 -input_height=1000 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail to gen bs1 result!"
    exit -1
fi
rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=16 -om_path=retinaface_bs16.om -input_text_path=./retinaface_pre_bin.info -input_width=1000 -input_height=1000 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail to gen bs16 result!!"
    exit -1
fi

python3.7 retinaface_pth_postprocess.py --result-folder=${currentDir}/result_write_om1 --out-folder=${currentDir}/result/dumpOutput_device0 --info-folder=${currentDir}/widerface/prep_info
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python3.7 retinaface_pth_postprocess.py --result-folder=${currentDir}/result_write_om16 --out-folder=${currentDir}/result/dumpOutput_device1 --info-folder=${currentDir}/widerface/prep_info
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

cd ${currentDir}/Pytorch_Retinaface/widerface_evaluate
if [ -f "bbox.cpython-37m-x86_64-linux-gnu.so" ]; then
    echo 'no need to build'
else
    python3.7 setup.py build_ext --inplace
fi
python3.7 evaluation.py -p=${currentDir}/result_write_om1/ > ${currentDir}/acc_bs1.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 evaluation.py -p=${currentDir}/result_write_om16/ > ${currentDir}/acc_bs16.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data om1===="
cat ${currentDir}/acc_bs1.txt
echo "====accuracy data om16===="
cat ${currentDir}/acc_bs16.txt

echo "====performance data om1===="
str=`grep "Interface" ${currentDir}/result/perf_vision_batchsize_1_device_0.txt`
perf_num=`echo $str | awk -F' ' '{print $6}' | awk -F',' '{print $1}'`
awk 'BEGIN{printf "310 bs1 fps:%.3f\n", '$perf_num'*4}'
echo "====performance data om16===="
str=`grep "Interface" ${currentDir}/result/perf_vision_batchsize_16_device_1.txt`
perf_num=`echo $str | awk -F' ' '{print $6}' | awk -F',' '{print $1}'`
awk 'BEGIN{printf "310 bs16 fps:%.3f\n", '$perf_num'*4}'
echo "success"     