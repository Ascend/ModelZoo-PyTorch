#!/bin/bash

#网络名称,同目录名称,需要模型审视修改
Network="clip"

model_name=clip
train_epochs=3
batch_size=64
model_path=""
data_path=""
device_id=0

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
  fi
done

#校验是否传入model_path,不需要修改
if [[ $model_path == "" ]]; then
  echo "[Error] para \"model_path\" must be confing"
  exit 1
fi
#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]]; then
  echo "[Error] para \"data_path\" must be confing"
  exit 1
fi
# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi

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
#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ]; then
  rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
  mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/
else
  mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/
fi

source ${cur_path}/test/env_npu.sh

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

if [ $(uname -m) = "aarch64" ]; then
  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT=29500
  export WORLD_SIZE=8
  for i in $(seq 0 7); do
    export RANK=${i}
    let p_start=0+24*i
    let p_end=23+24*i
    taskset -c $p_start-$p_end $CMD \
      python3 ./run_clip.py --output_dir ./clip-roberta-finetuned-npu-8p \
      --num_train_epochs ${train_epochs} \
      --model_name_or_path "$model_path" \
      --data_dir $data_path \
      --dataset_name ydshieh/coco_dataset_script \
      --dataset_config_name=2017 \
      --image_column image_path --caption_column caption \
      --remove_unused_columns=False \
      --do_train --do_eval --fp16 --dataloader_drop_last \
      --dataloader_num_workers 8 \
      --fp16_opt_level O2 --loss_scale 12800000 --optim adamw_apex_fused_npu --use_combine_grad \
      --per_device_train_batch_size=$batch_size --per_device_eval_batch_size=$batch_size \
      --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
      --save_steps 15000 --skip_steps 10 \
      --overwrite_output_dir \
      --local_rank $i >${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}.log 2>&1 &
  done
else
  python3 -m torch.distributed.launch --nproc_per_node 8 --use_env \
    ./run_clip.py --output_dir ./clip-roberta-finetuned-npu-8p \
    --num_train_epochs ${train_epochs} \
    --model_name_or_path "$model_path" \
    --data_dir $data_path \
    --dataset_name ydshieh/coco_dataset_script \
    --dataloader_num_workers 8 \
    --dataset_config_name=2017 \
    --image_column image_path --caption_column caption \
    --remove_unused_columns=False \
    --do_train --do_eval --fp16 --dataloader_drop_last \
    --fp16_opt_level O2 --loss_scale 12800000 --optim adamw_apex_fused_npu --use_combine_grad \
    --per_device_train_batch_size=$batch_size --per_device_eval_batch_size=$batch_size \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --save_steps 15000 --skip_steps 10 \
    --ddp_bucket_cap_mb 150 \
    --overwrite_output_dir >${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}.log 2>&1 &
fi

wait

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
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'acc'

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
