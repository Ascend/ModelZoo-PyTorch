#!/usr/bin/env

Network="chinese_clip"
WORK_DIR=""
LOAD_FROM=""

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

start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=$(env | grep etp_running_flag)
etp_flag=$(echo ${check_etp_flag#*=})
if [ x"${etp_flag}" != x"true" ]; then
  source ${test_path_dir}/env_npu.sh
fi

# Number of GPUs per GPU worker
GPUS_PER_NODE=1
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
# export MASTER_ADDR=XX.XX.XX.XX
export MASTER_ADDR=localhost
# The port for communication
export MASTER_PORT=8514
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0 

export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip/

DATAPATH=${1}

# data options
train_data=${DATAPATH}/datasets/Flickr30k-CN/lmdb/train
val_data=${DATAPATH}/datasets/Flickr30k-CN/lmdb/valid # if val_data is not specified, the validation will be automatically disabled

# restore options
resume=${DATAPATH}/pretrained_weights/clip_cn_vit-b-16.pt # or specify your customed ckpt path to resume
reset_data_offset="--reset-data-offset"
reset_optimizer="--reset-optimizer"
# reset_optimizer=""

# output options
output_base_dir=${DATAPATH}/experiments/
name=flickr30k_finetune_vit-b-16_roberta-base_bs128_1gpu
save_step_frequency=999999 # disable it
save_epoch_frequency=1
log_interval=1
report_training_batch_acc="--report-training-batch-acc"
# report_training_batch_acc=""

# training hyper-params
context_length=52
warmup=100
batch_size=128
valid_batch_size=128
accum_freq=1
lr=5e-5
wd=0.001
max_epochs=3  # or you can alternatively specify --max-steps
valid_step_interval=999999
valid_epoch_interval=1
vision_model=ViT-B-16
text_model=RoBERTa-wwm-ext-base-chinese
use_augment="--use-augment"
# use_augment=""

WORLD_SIZE=1

python3 -m torch.distributed.launch --use_env --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} \
          --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} cn_clip/training/main.py \
          --train-data=${train_data} \
          --val-data=${val_data} \
          --resume=${resume} \
          ${reset_data_offset} \
          ${reset_optimizer} \
          --logs=${output_base_dir} \
          --name=${name} \
          --save-step-frequency=${save_step_frequency} \
          --save-epoch-frequency=${save_epoch_frequency} \
          --log-interval=${log_interval} \
          ${report_training_batch_acc} \
          --context-length=${context_length} \
          --warmup=${warmup} \
          --batch-size=${batch_size} \
          --valid-batch-size=${valid_batch_size} \
          --valid-step-interval=${valid_step_interval} \
          --valid-epoch-interval=${valid_epoch_interval} \
          --accum-freq=${accum_freq} \
          --lr=${lr} \
          --wd=${wd} \
          --max-epochs=${max_epochs} \
          --vision-model=${vision_model} \
          ${use_augment} \
          --text-model=${text_model} \
          >$cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))


# 训练用例信息，不需要修改
BatchSize=${batch_size}*${WORLD_SIZE}
DeviceType=$(uname -m)
CaseName=${Network}_bs${batch_size}_${WORLD_SIZE}'p'_'acc'

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
avg_time=`grep -a 'Batch Time'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "Batch Time: " '{print $2}'|awk -F "s" '{print $1}'|tail -100 | awk '{a+=$1} END {if (NR != 0) printf("%.3f",a/NR)}'`
FPS=`echo "scale=2;$BatchSize / $avg_time" |bc`
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

TrainingTime=`grep -a 'Batch Time'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "Batch Time: " '{print $2}'|awk -F "s" '{print $1}'| awk '{a+=$1} END {printf("%.3f",a)}'`


bash ${test_path_dir}/flickr30k_evaluation_vit-b-16_rbt-base_1p.sh .

text_to_image_acc_r5=`awk -F '\"r5\":' '{print $2}' output.json | awk -F ',' '{print $1}'`
image_to_text_acc_r5=`awk -F '\"r5\":' '{print $2}' output_tr.json | awk -F ',' '{print $1}'`

echo "Network = ${Network}" >${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${batch_size}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "text_to_image_acc_r5 = ${text_to_image_acc_r5}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "image_to_text_acc_r5 = ${image_to_text_acc_r5}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${FPS}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}_perf_report.log
echo "TrainingTime = ${TrainingTime}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}_perf_report.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}_perf_report.log