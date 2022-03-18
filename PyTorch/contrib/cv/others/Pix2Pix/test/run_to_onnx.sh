#!/bin/bash


# bash ./test/run_to_onnx.sh --dataroot  --pth_path=./checkpoints/facades_pix2pix_npu_1p_full



for para in $*
do
    if [[ $para == --pth_path* ]];then
        pth_path=`echo ${para#*=}`
    fi
done

cur_path=`pwd`
cur_path_last_diename=${cur_path##*/}
if [ x"${cur_path_last_diename}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

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
    --dataroot ./datasets/facades/ \
    --checkpoints_dir ${checkpoints_dir} \
    --norm instance \
    --name ${name} \
