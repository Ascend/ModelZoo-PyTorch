#!/bin/bash
################基础配置参数，需要模型审视修改##################
# 网络名称，同目录名称
Network="GLIP_for_PyTorch"
BATCH_SIZE=8
TEST_BATCH_SIZE=8
USE_AMP=False
early_stop_iteration=2500
LOAD_FROM="pretrain/glip_tiny_model_o365_goldg_cc_sbu.pth"
TOKENIZER_PATH="./bert-base-uncased"
MODEL_PATH="./bert-base-uncased"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
export WORLD_SIZE=8
export MASTER_PORT=29500
export MASTER_ADDR=127.0.0.1

for para in $*
do
    if [[ $para == --batch_size* ]]; then
        BATCH_SIZE=$(echo ${para#*=})
    elif [[ $para == --load_from* ]]; then
        LOAD_FROM=$(echo ${para#*=})
    elif [[ $para == --fp32* ]]; then
        USE_AMP=False
    elif [[ $para == --fp16* ]]; then
        USE_AMP=True
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

RANK_ID_START=0
RANK_SIZE=8
KERNEL_NUM=$(($(nproc)/8))
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    export RANK=$RANK_ID
    export LOCAL_RANK=$RANK_ID
    PID_START=$((KERNEL_NUM * RANK_ID))
    PID_END=$((PID_START + KERNEL_NUM - 1))
    nohup taskset -c $PID_START-$PID_END python tools/train_net.py \
        --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
        --skip-test \
        --local_rank $RANK_ID \
        --override_output_dir ${WORK_DIR} \
        --early_stop_iteration ${early_stop_iteration} \
        MODEL.WEIGHT ${LOAD_FROM} \
        MODEL.LANGUAGE_BACKBONE.TOKENIZER_PATH ${TOKENIZER_PATH} \
        MODEL.LANGUAGE_BACKBONE.MODEL_PATH ${MODEL_PATH} \
        DATASETS.TRAIN '("coco_grounding_train", )' \
        MODEL.BACKBONE.FREEZE_CONV_BODY_AT -1 SOLVER.IMS_PER_BATCH ${BATCH_SIZE} \
        SOLVER.USE_AMP ${USE_AMP} SOLVER.MAX_EPOCH 24 \
        TEST.DURING_TRAINING False TEST.IMS_PER_BATCH ${TEST_BATCH_SIZE} \
        TEST.AFTER_TRAINING True SOLVER.TEST_WITH_INFERENCE True \
        SOLVER.FIND_UNUSED_PARAMETERS False SOLVER.BASE_LR 0.00001 \
        SOLVER.LANG_LR 0.00001 SOLVER.STEPS \(0.67,0.89\) \
        DATASETS.DISABLE_SHUFFLE True MODEL.DYHEAD.SCORE_AGG "MEAN" \
        TEST.EVAL_TASK detection \
        >>$cur_path/test/output/${ASCEND_DEVICE_ID}/train_${RANK_ID}.log 2>&1 &
done
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
# 输出性能FPS，需要模型审视修改
FPS=`grep -a 'fps' ${WORK_DIR}/train_${ASCEND_DEVICE_ID}.log|awk -F "fps: " '{print $2}'|awk -F "(" '{print $1}'| tail -2400 | awk '{a+=$1} END {if (NR != 0) printf("%.3f",a/NR)}'`
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

# 输出训练精度,需要模型审视修改
train_accuracy=$(grep -a ".inference INFO: OrderedDict" ${WORK_DIR}/train_${ASCEND_DEVICE_ID}.log|tail -1 |awk -F "'AP', " '{print $2}' |awk -F ")" '{print $1}' |awk '{a+=$1} END {printf("%.3f",a)}')
# 打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"


# 性能看护结果汇总
# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 训练总时长
TrainingTime=`grep -a 'time'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "time: " '{print $2}'|awk -F "," '{print $1}'| awk '{a+=$1} END {printf("%.3f",a)}'`

# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "time" ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F "loss: " '{print $2}'|awk -F "(" '{print $1}' >>${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

# # 最后一个迭代loss值，不需要修改
# ActualLoss=$(awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt)

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}_perf_report.log
echo "TrainingTime = ${TrainingTime}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}_perf_report.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}_perf_report.log