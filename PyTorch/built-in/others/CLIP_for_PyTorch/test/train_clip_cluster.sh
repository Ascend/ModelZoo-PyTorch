#!/bin/bash

Network="clip"
model_name=clip
train_epochs=3
batch_size=64
model_path=""
data_path=""
nnodes=""
node_rank=""
master_addr=""
master_port=""

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
#集合通信参数,不需要修改
export RANK_SIZE=8

# 数据集路径,保持为空,不需要修改
data_path=""
#网络名称
device_number=8
for para in $*; do
  if [[ $para == --model_name* ]]; then
    model_name=$(echo ${para#*=})
  elif [[ $para == --batch_size* ]]; then
    batch_size=$(echo ${para#*=})
  elif [[ $para == --model_path* ]]; then
    model_path=$(echo ${para#*=})
  elif [[ $para == --data_path* ]]; then
    data_path=$(echo ${para#*=})
  elif [[ $para == --train_epochs* ]]; then
    train_epochs=$(echo ${para#*=})
  elif [[ $para == --node_rank* ]]; then
    node_rank=$(echo ${para#*=})
  elif [[ $para == --master_addr* ]]; then
    master_addr=$(echo ${para#*=})
  elif [[ $para == --master_port* ]]; then
    master_port=$(echo ${para#*=})
  elif [[ $para == --nnodes* ]]; then
    nnodes=$(echo ${para#*=})
  fi
done

#校验是否传入model_path
if [[ $model_path == "" ]]; then
  echo "[Error] para \"model_path\" must be confing"
  exit 1
fi
#校验是否传入data_path
if [[ $data_path == "" ]]; then
  echo "[Error] para \"data_path\" must be confing"
  exit 1
fi
#校验是否传入nnodes
if [[ $nnodes == "" ]]; then
  echo "[Error] para \"nnodes\" must be confing"
  exit 1
fi
#校验是否传入node_rank
if [[ $node_rank == "" ]]; then
  echo "[Error] para \"node_rank\" must be confing"
  exit 1
fi
#校验是否传入master_addr
if [[ $master_addr == "" ]]; then
  echo "[Error] para \"master_addr\" must be confing"
  exit 1
fi
#校验是否传入master_port
if [[ $master_port == "" ]]; then
  echo "[Error] para \"master_port\" must be confing"
  exit 1
fi

#非平台场景时source 环境变量
check_etp_flag=$(env | grep etp_running_flag)
etp_flag=$(echo ${check_etp_flag#*=})
if [ x"${etp_flag}" != x"true" ]; then
  source ${test_path_dir}/env_npu.sh
fi

ASCEND_DEVICE_ID=0
#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ]; then
  rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
  mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
  mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=$(hostname -I | awk '{print $1}')
export MASTER_ADDR=${master_addr}
export MASTER_PORT=${master_port}
export WORLD_SIZE=$((nnodes * device_number))

#训练开始时间，不需要修改
start_time=$(date +%s)
if [ $(uname -m) = "aarch64" ]; then
  for i in $(seq 0 7); do
    rank=$((i + node_rank * device_number))
    export RANK=${rank}
    export LOCAL_RANK=$i
    KERNEL_NUM=$(($(nproc) / 8))
    PID_START=$((KERNEL_NUM * i))
    PID_END=$((PID_START + KERNEL_NUM - 1))
    taskset -c $PID_START-$PID_END $CMD \
      python3 ./run_clip.py --output_dir ./clip-roberta-finetuned-npu-8p \
      --num_train_epochs ${train_epochs} \
      --model_name_or_path "$model_path" \
      --data_dir $data_path \
      --dataset_name ydshieh/coco_dataset_script \
      --dataset_config_name=2017 \
      --dataloader_num_workers 8 \
      --image_column image_path --caption_column caption \
      --remove_unused_columns=False \
      --do_train --do_eval --fp16 --dataloader_drop_last \
      --fp16_opt_level O2 --loss_scale 12800000 --optim adamw_apex_fused_npu --use_combine_grad \
      --per_device_train_batch_size=$batch_size --per_device_eval_batch_size=$batch_size \
      --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
      --overwrite_output_dir \
      --save_steps 15000 --skip_steps 10 \
      --local_rank $i >${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}.log 2>&1 &
  done
else
  python3 -m torch.distributed.launch --nproc_per_node 8 \
    --nnodes=$nnodes \
    --node_rank $node_rank \
    --master_addr $master_addr \
    --master_port $master_port \
    ./run_clip.py --output_dir ./clip-roberta-finetuned-npu-8p \
    --num_train_epochs ${train_epochs} \
    --model_name_or_path "$model_path" \
    --data_dir $data_path \
    --dataset_name ydshieh/coco_dataset_script \
    --dataset_config_name=2017 \
    --dataloader_num_workers 8 \
    --image_column image_path --caption_column caption \
    --remove_unused_columns=False \
    --do_train --do_eval --fp16 --dataloader_drop_last \
    --fp16_opt_level O2 --loss_scale 12800000 --optim adamw_apex_fused_npu --use_combine_grad \
    --per_device_train_batch_size=$batch_size --per_device_eval_batch_size=$batch_size \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --save_steps 15000 --skip_steps 10 \
    --overwrite_output_dir >${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}.log 2>&1 &
fi

wait

total_device=$((nnodes * device_number))
if [[ x"${master_addr}" == x"${HCCL_IF_IP}" ]]; then
  #训练结束时间，不需要修改
  end_time=$(date +%s)
  e2e_time=$(($end_time - $start_time))

  #结果打印，不需要修改
  echo "------------------ Final result ------------------"
  #输出性能FPS，需要模型审视修改
  FPS=$(grep "train_samples_per_second =" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $3}')

  #打印，不需要修改
  echo "Final Performance images/sec : $FPS"

  #输出训练精度,需要模型审视修改
  train_loss=$(grep "eval_loss" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $3}')

  #打印，不需要修改
  echo "Final Train Loss : ${train_loss}"
  echo "E2E Training Duration sec : $e2e_time"

  #性能看护结果汇总
  #训练用例信息，不需要修改
  BatchSize=${batch_size}
  DeviceType=$(uname -m)
  CaseName=${Network}_bs${BatchSize}_${total_device}'p'_'perf'

  ##获取性能数据，不需要修改
  #吞吐量
  ActualFPS=${FPS}
  #单迭代训练时长
  TrainingTime=$(awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}')

  #从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
  grep "{'loss" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "{'loss" '{print $2}' | awk '{print $2}' | awk -F "," '{print $1}' >${test_path_dir}/output/${ASCEND_DEVICE_ID}//train_${CaseName}_loss.txt
  #最后一个迭代loss值，不需要修改
  ActualLoss=$(awk 'END {print}' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt)

  #关键信息打印到${CaseName}.log中，不需要修改
  echo "Network = ${Network}" >${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
  echo "RankSize = ${WORLD_SIZE}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
  echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
  echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
  echo "CaseName = ${CaseName}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
  echo "ActualFPS = ${ActualFPS}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
  echo "TrainingTime = ${TrainingTime}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
  echo "ActualLoss = ${ActualLoss}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
  echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
  echo "TrainAccuracy = ${train_loss}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
fi
