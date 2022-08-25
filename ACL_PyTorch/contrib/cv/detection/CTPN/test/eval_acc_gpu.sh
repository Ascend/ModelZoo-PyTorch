#!/bin/bash

datasets_path="./data/Challenge2_Test_Task12_Images"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

cd data
rm -rf pth_txt
mkdir pth_txt
cd ..
python3.7 ctpn_postprocess.py --model=pth --imgs_dir=${datasets_path} --pth_txt=data/pth_txt
if [ $? != 0 ]; then
    echo "post process fail!"
    exit -1
fi

rm -rf script/pth_txt.zip
cd data/pth_txt
zip -rq pth_txt.zip ./*
mv pth_txt.zip ../../script/
cd ../..
python3.7 script/script.py -g=script/gt.zip –s=script/pth_txt.zip > data/pth_result.json

echo "====pth accuracy data===="
python3.7 test/parse.py data/pth_result.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "success"