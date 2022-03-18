#!/bin/bash

################基础配置参数##################
# 网络名称，同目录名称
Network="DQN"
# 训练使用的npu卡数
export RANK_SIZE=1
# 指定训练所使用的npu device卡id
device_id=0

# 校验是否指定了device_id,分动态分配device_id与手动指定device_id
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#################创建日志输出目录#################
 if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
     rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
     mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
 else
     mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
 fi

 #################启动训练脚本#################
 #训练开始时间
 start_time=$(date +%s)

 #source 环境变量
source test/env_npu.sh

pth_path=''
status_path=''

for para in $*
do
    if [[ $para == --pth_path* ]];then
        pth_path=`echo ${para#*=}`
    elif [[ $para == --status_path* ]];then
        status_path=`echo ${para#*=}`
    fi
done
taskset -c 0-32
python3 train_dqn.py --use_device='use_npu' \
            --device_id=0 \
            --max_step=1000 \
            --log_interval=100 \
            --eval_interval=1000 \
            --tag='train_eval_1p' \
            --pth_path=${pth_path} \
            --status_path=${status_path}


##################获取训练数据################
#训练结束时间
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印
echo "------------------ Final result ------------------"
#输出训练精度
episodic_return_test=`grep -a 'INFO: steps' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_eval_1p_${ASCEND_DEVICE_ID}.txt | tail -1 | awk -F " " '{print $10}'`
#打印
echo "Final Train Accuracy : ${episodic_return_test}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息
DeviceType=`uname -m`
CaseName=${Network}_bs_${RANK_SIZE}'p'_'acc'

#关键信息打印到${CaseName}.log中
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "EpisodicReturnTest = ${episodic_return_test}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log