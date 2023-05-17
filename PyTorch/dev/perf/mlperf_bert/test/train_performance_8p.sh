#!/bin/bash

#当前路径,不需要修改
cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}/test
fi
source ${test_path_dir}/env_npu.sh

#网络名称,同目录名称,需要模型审视修改
Network="mlperf_bert"

# 指定训练所使用的npu device卡id
device_id=0
# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
if [ ${device_id} ]; then
  export ASCEND_DEVICE_ID=${device_id}
  echo "device id is ${ASCEND_DEVICE_ID}"
else
  "[Error] device id must be config"
  exit 1
fi

#创建DeviceID输出目录，不需要修改
if [ -d $test_path_dir/output/${ASCEND_DEVICE_ID} ]; then
  rm -rf $test_path_dir/output/$ASCEND_DEVICE_ID
  mkdir -p $test_path_dir/output/$ASCEND_DEVICE_ID
else
  mkdir -p $test_path_dir/output/$ASCEND_DEVICE_ID
fi

#################启动训练脚本#################
data_and_model_path=""
batch_size=10
loss_scale=0
fp16=0
#获取外部传参，可扩展
for para in $*
do
  if [[ $para == --data_and_model_path* ]];then
    data_and_model_path=`echo ${para#*=}`
  elif [[ $para == --batch_size* ]];then
    batch_size=`echo ${para#*=}`
  elif [[ $para == --loss_scale* ]];then
    loss_scale=`echo ${para#*=}`
  elif [[ $para == --fp16* ]];then
    fp16=`echo ${para#*=}`
  fi
done

start_time=$(date +%s)

if [[ fp16 -ne 0 ]]; then
  nohup python3 -m torch.distributed.launch --nproc_per_node=8 run.py \
    --input_dir=$data_and_model_path/2048_shards_uncompressed \
    --eval_dir=$data_and_model_path/eval_set_uncompressed \
    --learning_rate=2e-5 \
    --bert_model="bert-large-uncased" \
    --output_dir='./checkpoint/' \
    --train_mlm_accuracy_window_size=0 \
    --warmup_proportion=0 \
    --warmup_steps=0 \
    --start_warmup_step=0 \
    --target_mlm_accuracy=0.720 \
    --weight_decay_rate=0.01 \
    --eval_iter_start_samples=125000 \
    --eval_iter_samples=125000 \
    --max_seq_length=512 \
    --cache_eval_data \
    --max_steps=100 \
    --max_predictions_per_seq=76 \
    --train_batch_size=${batch_size} \
    --eval_batch_size=${batch_size} \
    --max_samples_termination=14000000 \
    --gradient_accumulation_steps=1 \
    --do_train \
    --fp16 \
    --loss_scale=${loss_scale} \
    --bert_config_path=$data_and_model_path/bert_config.json \
    --dense_seq_outpu \
    --init_checkpoint=$data_and_model_path/model.ckpt-28252.pt > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
else
  nohup python3 -m torch.distributed.launch --nproc_per_node=8 run.py \
    --input_dir=$data_and_model_path/2048_shards_uncompressed \
    --eval_dir=$data_and_model_path/eval_set_uncompressed \
    --learning_rate=2e-5 \
    --bert_model="bert-large-uncased" \
    --output_dir='./checkpoint/' \
    --train_mlm_accuracy_window_size=0 \
    --warmup_proportion=0 \
    --warmup_steps=0 \
    --start_warmup_step=0 \
    --target_mlm_accuracy=0.720 \
    --weight_decay_rate=0.01 \
    --eval_iter_start_samples=125000 \
    --eval_iter_samples=125000 \
    --max_seq_length=512 \
    --cache_eval_data \
    --max_steps=100 \
    --max_predictions_per_seq=76 \
    --train_batch_size=${batch_size} \
    --eval_batch_size=${batch_size} \
    --max_samples_termination=14000000 \
    --gradient_accumulation_steps=1 \
    --do_train \
    --bert_config_path=$data_and_model_path/bert_config.json \
    --dense_seq_outpu \
    --init_checkpoint=$data_and_model_path/model.ckpt-28252.pt > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
fi
wait

##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=$(grep "seq/s" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "seq/s':" '{print $2}' | awk -F "," '{print $1}' | tail -n 2 | head -n 1)

#打印，不需要修改
echo "Final Performance seq/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=$(grep "eval_mlm_accuracy" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "eval_mlm_accuracy':" '{print $2}' | awk -F "}" '{print $1}' | awk 'END {print $1}')

#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=64 #仅用于打印，不在训练中生效
WORLD_SIZE=1
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "loss:" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "loss:" '{print $2}'| awk '{print $1}' >${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log