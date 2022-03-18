#!/bin/bash

#datasets_path="../test"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $datasets_path == "" ]];then
    echo "[Error] para \"datasets_path\" must be confing"
    exit 1
fi


rm -rf preprocess_data

arch=`uname -m`
python3.7 srgan_preprocess.py --src_path=${datasets_path} --set5_only=True
if [ $? != 0 ]; then
    echo "preprocess fail!"
    exit -1
fi
python3.7 gen_dataset_info.py
if [ $? != 0 ]; then
    echo "gen dataset info fail!"
    exit -1
fi
source env.sh
rm -rf result/dumpOutput_device0
chmod +x benchmark.${arch}
python3.7 task_process.py
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf infer_om_res
mkdir infer_om_res

python3.7 srgan_om_infer.py --data_path=${datasets_path}/data --target_path=${datasets_path}/target --result_path=./result/dumpOutput_device0
if [ $? != 0 ]; then
    echo "post process fail!"
    exit -1
fi

#rm -rf script/predict_txt.zip
#cd data/predict_txt
#zip -rq predict_txt.zip ./*
#mv predict_txt.zip ../../script/
#cd ../..
#python3.7 script/script.py -g=script/gt.zip –s=script/predict_txt.zip > data/result.json

#echo "====310 accuracy data===="
#python3.7 test/parse.py data/result.json
#if [ $? != 0 ]; then
#    echo "fail!"
#    exit -1
#fi


echo "success"
