#!/bin/bash

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
DATASET="./data/my-gpt_text_sentence"
BASE_DATA_PATH=./data/

VOCAB_PATH=./data/gpt2-vocab.json
MERGE_FILE=./data/gpt2-merges.txt

CONFIG_JSON=./ds_config.json

USE_DEEPSPEED=1
ZERO_STAGE=0

TP=1
PP=2
HIDDEN=1024
LAYERS=24
SEQ=1024
GLOBAL_BATCH=16
WORKER_STR=""

MICRO_BATCH=4

CHECKPOINT_PATH=ckpts/ckpts_tmp

while [ $# -gt  0 ]
do
key="$1"
case $key in
    --no-deepspeed)
    USE_DEEPSPEED=0;
    shift
    ;;
    -z|--zero-stage)
    ZERO_STAGE=$2;
    shift
    ;;
    *)
    echo "Unknown argument(s)"
    usage
    exit 1
    shift
    ;;
esac
done


options=" \
	--tensor-model-parallel-size $TP \
	--pipeline-model-parallel-size $PP \
  --num-layers $LAYERS \
  --hidden-size $HIDDEN \
  --num-attention-heads 16 \
  --seq-length $SEQ \
  --max-position-embeddings $SEQ \
	--micro-batch-size $MICRO_BATCH \
	--global-batch-size $GLOBAL_BATCH \
	--train-iters 500000 \
  --lr-decay-iters 320000 \
  --save $CHECKPOINT_PATH \
  --data-path $DATASET \
  --vocab-file $VOCAB_PATH \
  --merge-file $MERGE_FILE \
  --data-impl mmap \
  --split 949,50,1 \
  --distributed-backend nccl \
  --lr 0.00015 \
  --lr-decay-style cosine \
  --min-lr 1.0e-5 \
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

python3 run.py --include localhost:0,1,2,3,4,5,6,7 ${DIR}/pretrain_gpt.py $@ ${options} &> ./training.log &
wait

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
IterTime=$(grep "elapsed time per iteration" ./training.log | tail -n 1000 | awk '{print $14}' | awk '{sum+=$1} END {print sum/NR}')
FPS=$(echo "${GlobalBatchSize} * 1000 / ${IterTime}"|bc)

#打印，不需要修改
echo "Final Performance samples/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=$(grep "PPL" ./training.log | awk 'END {print $15}')

#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
