#!/bin/bash
# bash ./pth2om.sh  --pth_path=./checkpoints/facades_label2photo_pretrained --soc_version=Ascend310P3

for para in $*
do
    if [[ $para == --pth_path* ]];then
        pth_path=`echo ${para#*=}`
    fi

    if [[ $para == --soc_version* ]];then
        soc_version=`echo ${para#*=}`
    fi
done


if [[ $pth_path == "" ]];then
    echo "[Error] para \"pth_path\" must be confing"
    exit 1
fi

if [[ $soc_version == "" ]];then
    echo "[Error] para \"soc_version\" must be confing"
    exit 1
fi

checkpoints_dir=${pth_path%/*}
name=${pth_path##*/}

python pix2pix_pth2onnx.py \
    --direction BtoA \
    --model pix2pix \
    --checkpoints_dir ${checkpoints_dir} \
    --name ${name} \

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


atc --framework=5 --model=./checkpoints/facades_label2photo_pretrained/netG_onnx.onnx --output=./checkpoints/facades_label2photo_pretrained/netG_om_bs1 --input_format=NCHW --input_shape="inputs:1,3,256,256" --log=debug --soc_version=${soc_version}
export TUNE_BANK_PATH=/home/lrb/knowledgebank/bs4
atc --framework=5 --model=./checkpoints/facades_label2photo_pretrained/netG_onnx.onnx --output=./checkpoints/facades_label2photo_pretrained/netG_om_bs4 --input_format=NCHW --input_shape="inputs:4,3,256,256" --log=debug --soc_version=${soc_version}
export TUNE_BANK_PATH=/home/lrb/knowledgebank/bs8
atc --framework=5 --model=./checkpoints/facades_label2photo_pretrained/netG_onnx.onnx --output=./checkpoints/facades_label2photo_pretrained/netG_om_bs8 --input_format=NCHW --input_shape="inputs:8,3,256,256" --log=debug --soc_version=${soc_version}
export TUNE_BANK_PATH=/home/lrb/knowledgebank/bs16
atc --framework=5 --model=./checkpoints/facades_label2photo_pretrained/netG_onnx.onnx --output=./checkpoints/facades_label2photo_pretrained/netG_om_bs16 --input_format=NCHW --input_shape="inputs:16,3,256,256" --log=debug --soc_version=${soc_version}
export TUNE_BANK_PATH=/home/lrb/knowledgebank/bs32
atc --framework=5 --model=./checkpoints/facades_label2photo_pretrained/netG_onnx.onnx --output=./checkpoints/facades_label2photo_pretrained/netG_om_bs32 --input_format=NCHW --input_shape="inputs:32,3,256,256" --log=debug --soc_version=${soc_version}
export TUNE_BANK_PATH=/home/lrb/knowledgebank/bs64
atc --framework=5 --model=./checkpoints/facades_label2photo_pretrained/netG_onnx.onnx --output=./checkpoints/facades_label2photo_pretrained/netG_om_bs64 --input_format=NCHW --input_shape="inputs:64,3,256,256" --log=debug --soc_version=${soc_version}

echo "success"

