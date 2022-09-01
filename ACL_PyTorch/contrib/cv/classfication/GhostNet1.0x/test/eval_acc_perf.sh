#!/bin/bash

datasets_path="/opt/npu/"
ais_infer_path = "/home/HwHiAiUser/ais_infer"

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf ./prep_dataset
python3.7 imagenet_torch_preprocess.py ghostnet ${datasets_path}/imageNet/val ./prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
source env.sh
python ${ais_infer_path}/ais_infer.py --model ./ghostnet_bs1.om --input ./prep_dataset/ --output ./ --outfmt NPY --batchsize 1
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python ${ais_infer_path}/ais_infer.py --model ./ghostnet_bs16.om --input ./prep_dataset/ --output ./ --outfmt NPY --batchsize 16
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 imagenet_acc_eval.py ./lcmout/2022_xx_xx-xx_xx_xx/sumary.json /home/HwHiAiUser/dataset/ImageNet/val_label.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 imagenet_acc_eval.py ./lcmout/2022_xx_xx-xx_xx_xx/sumary.json /home/HwHiAiUser/dataset/ImageNet/val_label.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"