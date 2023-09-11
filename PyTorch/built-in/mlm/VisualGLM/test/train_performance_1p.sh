#!/bin/bash
# 删除缓存文件

################基础配置参数，需要模型审视修改##################
# 网络名称，同目录名称
Network="VisualGLM"

# 配置环境变量
source env_npu.sh

# 基本参数
NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
MP_SIZE=1
WORLD_SIZE=1
WORK_DIR=""
LOAD_FROM=""
MODEL_TYPE="visualglm-6b"
MODEL_ARGS="--max_source_length 64 \
    --max_target_length 256 \
    --lora_rank 10 \ 
    --layer_range 0 14 \
    --pre_seq_len 4"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"


# 需要修改的参数
BATCH_SIZE=8
MAX_TRAIN_STEP=20000
train_data=$1   # dataset path
PRETRAIN_MODEL_PATH=$2
eval_data="./fewshot-data/dataset.json"   # 模型中未用到生成
export ASCEND_RT_VISIBLE_DEVICES=0  # 指定卡
ASCEND_DEVICE_ID=0           # 与上一行数字相同，用于生成文件夹
##


for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --pretrain_model* ]];then
        pretrain_model=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

if [[ $pretrain_model == "" ]];then
    echo "[Error] para \"pretrain_model\" must be confing"
    exit 1
fi

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


if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ]; then
  rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
  mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
else
  mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
fi

start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=$(env | grep etp_running_flag)
etp_flag=$(echo ${check_etp_flag#*=})
if [ x"${etp_flag}" != x"true" ]; then
  source ${test_path_dir}/env_npu.sh
fi

gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters $MAX_TRAIN_STEP \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${train_data} \
       --valid-data ${eval_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --save-interval 2000 \
       --eval-interval 50000 \
       --save "./checkpoints" \
       --split 1 \
       --eval-iters 10 \
       --eval-batch-size 8 \
       --zero-stage 1 \
       --lr 0.0001 \
       --batch-size $BATCH_SIZE \
       --skip-init \
       --fp16 \
       --pretrain_model_path $PRETRAIN_MODEL_PATH\
       --use_lora >$cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
"
            
run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port 16666 --hostfile ${HOST_FILE_PATH} finetune_visualglm.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}
set +x
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
avg_time=`grep -a 'Time'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "Time: " '{print $2}'|awk -F "," '{print $1}'|tail -100 | awk '{a+=$1} END {if (NR != 0) printf("%.3f",a/NR)}'`
FPS=`echo "$BatchSize / $avg_time" |bc`
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

# 输出训练精度,需要模型审视修改
train_accuracy=$(grep -a "Eval" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log  |tail -1|awk -F "image_to_text_R@5: " '{print $2}' | awk -F "image_to_text_R" '{print $1}')
# 打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"


# 性能看护结果汇总
# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 训练总时长
TrainingTime=`grep -a 'Time'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "Time: " '{print $2}'|awk -F "," '{print $1}'| awk '{a+=$1} END {printf("%.3f",a)}'`

# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "Time" ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F "Loss: " '{print $2}' >>${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
