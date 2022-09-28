#!/bin/bash
source test/env.sh

arch=`uname -m`
currentDir=$(cd "$(dirname "$0")";pwd)/..

echo ----------------------------dataset preprocess------------------------------
rm -rf ./widerface
python3.7 retinaface_pth_preprocess.py
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo ----------------------------om model offline infer------------------------------
rm -rf result_bs1
mkdir result_bs1
python3.7 ./ais_infer/ais_infer.py --model retinaface_bs1.om --device 0 --batchsize 1 --input ./widerface/prep/ --output ./result_bs1 --outfmt BIN
if [ $? != 0 ]; then
    echo "fail to infer bs1 result!"
    exit -1
fi
rm -rf result_bs16
mkdir result_bs16
python3.7 ./ais_infer/ais_infer.py --model retinaface_bs16.om --device 1 --batchsize 16 --input ./widerface/prep/ --output ./result_bs16 --outfmt BIN
if [ $? != 0 ]; then
    echo "fail to infer bs16 result!!"
    exit -1
fi

echo ----------------------------infer result post precess------------------------------
#bs1
mv ${currentDir}/result_bs1/20*/sumary.json ${currentDir}/result_bs1/
python3.7 retinaface_pth_postprocess.py --prediction-folder=${currentDir}/result_bs1/20*/ --info-folder=${currentDir}/widerface/prep_info --output-folder=${currentDir}/widerface_result_bs1
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
#bs16
mv ${currentDir}/result_bs16/20*/sumary.json ${currentDir}/result_bs16/
python3.7 retinaface_pth_postprocess.py --prediction-folder=${currentDir}/result_bs16/20*/ --info-folder=${currentDir}/widerface/prep_info --output-folder=${currentDir}/widerface_result_bs16
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo ----------------------------calculation result accuracy------------------------------
cd ${currentDir}/Pytorch_Retinaface/widerface_evaluate
if [ -f "bbox.cpython-37m-x86_64-linux-gnu.so" ]; then
    echo 'no need to build'
else
    python3.7 setup.py build_ext --inplace
fi
python3.7 evaluation.py -p=${currentDir}/widerface_result_bs1/ > ${currentDir}/acc_bs1.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 evaluation.py -p=${currentDir}/widerface_result_bs16/ > ${currentDir}/acc_bs16.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo ----------------------------display accuracy result----------------------------
echo "====accuracy data om1===="
cat ${currentDir}/acc_bs1.txt
echo "====accuracy data om16===="
cat ${currentDir}/acc_bs16.txt

echo ----------------------------display performance result----------------------------
echo "====performance data om1===="
perf_num=`awk -F'"throughput": ' '{print $2}' ${currentDir}/result_bs1/sumary.json | awk -F'}' '{print $1}'`
awk 'BEGIN{printf "310P3 bs1 fps:%.3f\n", '$perf_num'}'

echo "====performance data om16===="
perf_num=`awk -F'"throughput": ' '{print $2}' ${currentDir}/result_bs16/sumary.json | awk -F'}' '{print $1}'`
awk 'BEGIN{printf "310P3 bs16 fps:%.3f\n", '$perf_num'}'

echo "success"     