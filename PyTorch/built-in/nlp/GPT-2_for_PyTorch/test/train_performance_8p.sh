#!/bin/bash

# 定义可选的模型大小
SIZES=("345M" "1.3B" "2.7B" "3.7B")

model_size="345M"
data_path=""

for para in $*
do
    if [[ $para == --model_size* ]];then
        model_size=`echo ${para#*=}`
    fi
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

# 检查模型大小是否合法
if [[ ! " ${SIZES[@]} " =~ " ${model_size} " ]]; then
  echo "Invalid model size. Please choose from: ${SIZES[*]}"
  exit 1
else
  MODEL_SIZE=$model_size
fi

# 如果./data目录不存在，则创建软链接
if [ ! -d "./data" ]; then
  ln -snf "$data_path" ./data
fi

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
DATASET="./data/my-gpt_text_sentence"
BASE_DATA_PATH=./data/

VOCAB_PATH=./data/gpt2-vocab.json
MERGE_FILE=./data/gpt2-merges.txt

CONFIG_JSON=./ds_config.json

USE_DEEPSPEED=1
ZERO_STAGE=0

case $MODEL_SIZE in
  "345M")
    echo "Running 345M model..."
    TP=1
    PP=2
    HIDDEN=1024
    LAYERS=24
    SEQ=1024
    GLOBAL_BATCH=16
    MICRO_BATCH=4
    NUM_ATTN_HEADS=16
    LR=1.5e-4
    MIN_LR=1.5e-5
    export FusedAdam=1
    ;;
  "1.3B")
    echo "Running 1.3B model..."
    TP=2
    PP=2
    HIDDEN=2048
    LAYERS=24
    SEQ=1024
    GLOBAL_BATCH=32
    MICRO_BATCH=8
    NUM_ATTN_HEADS=16
    LR=2e-4
    MIN_LR=2e-5
    export FusedAdam=1
    ;;
  "2.7B")
    echo "Running 2.7B model..."
    TP=2
    PP=2
    HIDDEN=2560
    LAYERS=32
    SEQ=2048
    GLOBAL_BATCH=32
    MICRO_BATCH=8
    NUM_ATTN_HEADS=32
    LR=1.6e-4
    MIN_LR=1.6e-5
    ;;
  "3.7B")
    echo "Running 3.7B model..."
    TP=2
    PP=2
    HIDDEN=3072
    LAYERS=32
    SEQ=2048
    GLOBAL_BATCH=16
    MICRO_BATCH=4
    NUM_ATTN_HEADS=32
    LR=1.2e-4
    MIN_LR=1.2e-5
    ;;
esac

CHECKPOINT_PATH=ckpts/ckpts_tmp

options=" \
	--tensor-model-parallel-size $TP \
	--pipeline-model-parallel-size $PP \
  --num-layers $LAYERS \
  --hidden-size $HIDDEN \
  --num-attention-heads $NUM_ATTN_HEADS \
  --seq-length $SEQ \
  --max-position-embeddings $SEQ \
	--micro-batch-size $MICRO_BATCH \
	--global-batch-size $GLOBAL_BATCH \
	--train-iters 300 \
  --lr-decay-iters 300 \
  --save $CHECKPOINT_PATH \
  --data-path $DATASET \
  --vocab-file $VOCAB_PATH \
  --merge-file $MERGE_FILE \
  --data-impl mmap \
  --split 949,50,1 \
  --distributed-backend nccl \
  --lr $LR \
  --lr-decay-style cosine \
  --min-lr $MIN_LR \
  --num-workers 16 \
  --weight-decay 1e-2 \
  --clip-grad 1.0 \
  --lr-warmup-fraction .01 \
  --log-interval 100 \
  --save-interval 100000 \
  --eval-interval 1000 \
  --eval-iters 10 \
  --fp16 \
	--tensorboard-dir $CHECKPOINT_PATH/tensorboard_dir \
	--seed 1234 \
	--attention-dropout 0.0 \
	--hidden-dropout 0.0 \
  --checkpoint-activations \
        "

if [ ${USE_DEEPSPEED} -eq 1 ]; then
	echo "Using DeepSpeed"
	options="${options} \
		--deepspeed \
		--deepspeed_config=${CONFIG_JSON} \
		--zero-stage=${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
	"
fi

cat <<EOT > $CONFIG_JSON
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },

  "gradient_clipping": 1.0,
  "prescale_gradients": true,

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : false
}
EOT

source ./test/env_npu.sh

if [ ! -d "./test/output/0/" ]; then
  mkdir -p ./test/output/0/
fi

# 开始训练
start_time=$(date +%s)
python3 run.py --include localhost:0,1,2,3,4,5,6,7 ${DIR}/pretrain_gpt.py ${options} &> ./test/output/0/train_0.log &
wait

# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep "samples/sec" ./test/output/0/train_0.log | tail -n 100 | awk '{print $10}' | awk '{sum+=$1} END {print sum/NR}' `

#打印，不需要修改
echo "Final Performance samples/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=` grep "validation loss at iteration" ./test/output/0/train_0.log | awk '{print $15}' | awk 'BEGIN {min = 65536} {if ($1+0 < min+0) min=$1} END {print min}' `

#打印，不需要修改
echo "Final Train lm loss PPL(accuracy): ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"
