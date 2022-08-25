#!/bin/bash

datasets_path="./data/Challenge2_Test_Task12_Images"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

python3.7 task_process.py --mode='preprocess' --src_dir=${datasets_path}
if [ $? != 0 ]; then
    echo "preprocess fail!"
    exit -1
fi

rm -r result
mkdir result
cd result
mkdir inf_output
mkdir dumpOutput_device0
cd ..

python3.7 task_process.py --mode='ais_infer' --machine='310'
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

cd data
rm -rf predict_txt
mkdir predict_txt
cd ..
python3.7 ctpn_postprocess.py --imgs_dir=${datasets_path} --bin_dir=result/dumpOutput_device0 --predict_txt=data/predict_txt
if [ $? != 0 ]; then
    echo "post process fail!"
    exit -1
fi

rm -rf script/predict_txt.zip
cd data/predict_txt
zip -rq predict_txt.zip ./*
mv predict_txt.zip ../../script/
cd ../..
python3.7 script/script.py -g=script/gt.zip –s=script/predict_txt.zip > data/result.json

echo "====310 accuracy data===="
python3.7 test/parse.py data/result.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


echo "success"