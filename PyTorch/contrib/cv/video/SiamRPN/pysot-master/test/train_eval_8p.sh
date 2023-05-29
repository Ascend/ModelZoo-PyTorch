#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="SimRPN_for_PyTorch"


###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_diename=${cur_path##*/}
if [ x"${cur_path_last_diename}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#################创建日志输出目录，不需要修改#################
ASCEND_DEVICE_ID=0


# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi


python3  ${test_path_dir}/../tools_1p/test.py 	\
	--snapshot ${test_path_dir}/../snapshot_8p/checkpoint_e20.pth \
    --config ${test_path_dir}/../experiments/siamrpn_r50_l234_dwxcorr_8gpu/config.yaml \
	--datasetdir ${test_path_dir}/../../testing_dataset/VOT2016

wait


python3 ${test_path_dir}/../tools_1p/eval.py 	 \
	--tracker_path ${test_path_dir}/../results > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_acc_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

Acc=`grep -a 'acc' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_acc_${ASCEND_DEVICE_ID}.log | awk -F " " '{print $2}'|awk 'END {print}'`
Actualacc={$Acc}
echo "Fianl acc $Acc"


