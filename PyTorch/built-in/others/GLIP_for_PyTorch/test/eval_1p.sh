#!/bin/bash
################基础配置参数，需要模型审视修改##################
# 网络名称，同目录名称
Network="GLIP_for_PyTorch"
BATCH_SIZE=8
WORLD_SIZE=1
LOAD_FROM="pretrain/glip_tiny_model_o365_goldg_cc_sbu.pth"

for para in $*
do
    if [[ $para == --batch_size* ]]; then
        BATCH_SIZE=$(echo ${para#*=})
    elif [[ $para == --load_from* ]]; then
        LOAD_FROM=$(echo ${para#*=})
    fi
done

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}/test
fi

ASCEND_DEVICE_ID=0

if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ]; then
  rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
  mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
else
  mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
fi

WORK_DIR=${cur_path}/test/output/${ASCEND_DEVICE_ID}
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=$(env | grep etp_running_flag)
etp_flag=$(echo ${check_etp_flag#*=})
if [ x"${etp_flag}" != x"true" ]; then
  source ${test_path_dir}/env_npu.sh
fi

RANK_ID=0
KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))

taskset -c $PID_START-$PID_END python tools/test_grounding_net.py \
       --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
       --weight ${LOAD_FROM} \
       TEST.IMS_PER_BATCH ${BATCH_SIZE} \
       MODEL.DYHEAD.SCORE_AGG "MEAN" \
       TEST.EVAL_TASK detection \
       MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
       OUTPUT_DIR ${WORK_DIR} \
       >$cur_path/test/output/${ASCEND_DEVICE_ID}/eval_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 训练用例信息，不需要修改
BatchSize=${BATCH_SIZE}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'acc'

# 结果打印，不需要修改
echo "------------------ Final result ------------------"

# 输出训练精度,需要模型审视修改
train_accuracy=$(grep -a ".inference INFO: OrderedDict" ${WORK_DIR}/train_${ASCEND_DEVICE_ID}.log|tail -1 |awk -F "'AP', " '{print $2}' |awk -F ")" '{print $1}' |awk '{a+=$1} END {printf("%.3f",a)}')
# 打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

# 关键信息打印到${CaseName}.log中，不需要修改
echo "TrainAccuracy = ${train_accuracy}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
