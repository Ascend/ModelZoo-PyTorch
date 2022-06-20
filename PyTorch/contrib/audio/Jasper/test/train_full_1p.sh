#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="Jasper"
# 训练batch_size
batch_size=32
# 训练使用的npu卡数
export RANK_SIZE=1

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

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
echo ${pwd}

#################创建日志输出目录，不需要修改#################
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

# 变量
export DETECTRON2_DATASETS=${data_path}
export PYTHONPATH=./:$PYTHONPATH
export OMP_NUM_THREADS=1

: ${MODEL_CONFIG:=${2:-"configs/jasper10x5dr_speedp-online_speca.yaml"}}
: ${OUTPUT_DIR:=${3:-"/results"}}
: ${CHECKPOINT:=${4:-}}
: ${RESUME:=true}
: ${CUDNN_BENCHMARK:=true}
: ${NUM_GPUS:=1}
: ${AMP:=True}
: ${GRAD_ACCUMULATION_STEPS:=2}
: ${LEARNING_RATE:=0.01}
: ${MIN_LEARNING_RATE:=0.00001}
: ${LR_POLICY:=exponential}
: ${LR_EXP_GAMMA:=0.981}
: ${EMA:=0.999}
: ${SEED:=0}
: ${EPOCHS:=33}
: ${WARMUP_EPOCHS:=2}
: ${HOLD_EPOCHS:=140}
: ${SAVE_FREQUENCY:=10}
: ${EPOCHS_THIS_JOB:=0}
: ${DALI_DEVICE:="cpu"}
: ${PAD_TO_MAX_DURATION:=false}
: ${EVAL_FREQUENCY:=544}
: ${PREDICTION_FREQUENCY:=544}
: ${TRAIN_MANIFESTS:="$data_path/librispeech-train-clean-100-wav.json \
                      $data_path/librispeech-train-clean-360-wav.json \
                      $data_path/librispeech-train-other-500-wav.json"}
: ${VAL_MANIFESTS:="$data_path/librispeech-dev-clean-wav.json"}

mkdir -p "$OUTPUT_DIR"

ARGS="--dataset_dir=$data_path"
ARGS+=" --val_manifests $VAL_MANIFESTS"
ARGS+=" --train_manifests $TRAIN_MANIFESTS"
ARGS+=" --model_config=$MODEL_CONFIG"
ARGS+=" --output_dir=$OUTPUT_DIR"
ARGS+=" --lr=$LEARNING_RATE"
ARGS+=" --batch_size=$batch_size"
ARGS+=" --min_lr=$MIN_LEARNING_RATE"
ARGS+=" --lr_policy=$LR_POLICY"
ARGS+=" --lr_exp_gamma=$LR_EXP_GAMMA"
ARGS+=" --epochs=$EPOCHS"
ARGS+=" --warmup_epochs=$WARMUP_EPOCHS"
ARGS+=" --hold_epochs=$HOLD_EPOCHS"
ARGS+=" --epochs_this_job=$EPOCHS_THIS_JOB"
ARGS+=" --ema=$EMA"
ARGS+=" --seed=$SEED"
ARGS+=" --optimizer=novograd"
ARGS+=" --weight_decay=1e-3"
ARGS+=" --save_frequency=$SAVE_FREQUENCY"
ARGS+=" --keep_milestones 100 200 300 400"
ARGS+=" --save_best_from=380"
ARGS+=" --log_frequency=1"
ARGS+=" --eval_frequency=$EVAL_FREQUENCY"
ARGS+=" --prediction_frequency=$PREDICTION_FREQUENCY"
ARGS+=" --grad_accumulation_steps=$GRAD_ACCUMULATION_STEPS "
ARGS+=" --dali_device=$DALI_DEVICE"

[ "$AMP" = true ] &&                 ARGS+=" --amp"
[ "$RESUME" = true ] &&              ARGS+=" --resume"
[ "$CUDNN_BENCHMARK" = true ] &&     ARGS+=" --cudnn_benchmark"
[ -n "$MAX_DURATION" ] &&            ARGS+=" --override_config input_train.audio_dataset.max_duration=$MAX_DURATION" \
                                     ARGS+=" --override_config input_train.filterbank_features.max_duration=$MAX_DURATION"
[ "$PAD_TO_MAX_DURATION" = true ] && ARGS+=" --override_config input_train.audio_dataset.pad_to_max_duration=True" \
                                     ARGS+=" --override_config input_train.filterbank_features.pad_to_max_duration=True"
[ -n "$CHECKPOINT" ] &&              ARGS+=" --ckpt=$CHECKPOINT"
[ -n "$LOG_FILE" ] &&                ARGS+=" --log_file $LOG_FILE"
[ -n "$PRE_ALLOCATE" ] &&            ARGS+=" --pre_allocate_range $PRE_ALLOCATE"

#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

DISTRIBUTED="-m torch.distributed.launch --nproc_per_node=$NUM_GPUS"
python $DISTRIBUTED train.py $ARGS > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait


##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep "avg train utts/s" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk '{print $9}' | awk 'END {print}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep "avg" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | grep "wer" | awk '{print $16}' | awk 'END {print}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`


#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
