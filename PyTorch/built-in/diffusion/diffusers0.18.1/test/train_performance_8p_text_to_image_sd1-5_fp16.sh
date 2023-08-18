# 网络名称,同目录名称,需要模型审视修改
Network="diffusers"

# 预训练模型
model_name="runwayml/stable-diffusion-v1-5"
batch_size=1
max_train_steps=1500
mixed_precision="fp16"
resolution=512
dataset_name="lambdalabs/pokemon-blip-captions"
local_data_dir=""

for para in $*; do
  if [[ $para == --model_name* ]]; then
    model_name=$(echo ${para#*=})
  elif [[ $para == --batch_size* ]]; then
    batch_size=$(echo ${para#*=})
  elif [[ $para == --max_train_steps* ]]; then
    max_train_steps=$(echo ${para#*=})
  elif [[ $para == --mixed_precision* ]]; then
    mixed_precision=$(echo ${para#*=})
  elif [[ $para == --resolution* ]]; then
    resolution=$(echo ${para#*=})
  elif [[ $para == --dataset_name* ]]; then
    dataset_name=$(echo ${para#*=})
  elif [[ $para == --local_data_dir* ]]; then
    local_data_dir=$(echo ${para#*=})
  fi
done


NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

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

source ${test_path_dir}/env_npu.sh

#创建DeviceID输出目录，不需要修改
output_path=${cur_path}/test/output/${ASCEND_DEVICE_ID}

if [ -d ${output_path} ]; then
  rm -rf ${output_path}
fi

mkdir -p ${output_path}



#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

python3 -m torch.distributed.launch --nproc_per_node 8 --use_env \
  examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path=$model_name \
  --dataset_name=$dataset_name \
  --resolution=$resolution --center_crop --random_flip \
  --train_batch_size=$batch_size \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=$max_train_steps \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --checkpointing_steps=3000 \
  --use_ema \
  --mixed_precision=$mixed_precision \
  --dataloader_num_workers=8 \
  --use_megatron_npu_adamW \
  --output_dir=${test_path_dir}/output/$ASCEND_DEVICE_ID/  > ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#输出性能FPS，需要模型审视修改
FPS=$(grep "FPS: " ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $NF}')

#获取性能数据，不需要修改
#吞吐量
ActualFPS=$(awk 'BEGIN{printf "%.2f\n", '${FPS}'}')

#打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"

#loss值，不需要修改
ActualLoss=$(grep -o "step_loss=[0-9.]*" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $NF}')

#打印，不需要修改
echo "Final Train Loss : ${ActualLoss}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'acc'

#单迭代训练时长
TrainingTime=$(awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}')

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