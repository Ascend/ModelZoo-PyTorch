#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# bash ./test/pth2om.sh  --pth_path=./checkpoints/facades_label2photo_pretrained

for para in $*
do
    if [[ $para == --pth_path* ]];then
        pth_path=`echo ${para#*=}`
    fi
done


# 校验是否传入 pth_path , 验证脚本需要传入此参数
if [[ $pth_path == "" ]];then
    echo "[Error] para \"pth_path\" must be confing"
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

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

atc --framework=5 --model=./checkpoints/facades_label2photo_pretrained/netG_onnx.onnx --output=./checkpoints/facades_label2photo_pretrained/netG_om_bs1 --input_format=NCHW --input_shape="inputs:1,3,256,256" --log=debug --soc_version=Ascend310
atc --framework=5 --model=./checkpoints/facades_label2photo_pretrained/netG_onnx.onnx --output=./checkpoints/facades_label2photo_pretrained/netG_om_bs16 --input_format=NCHW --input_shape="inputs:16,3,256,256" --log=debug --soc_version=Ascend310

echo "success"
