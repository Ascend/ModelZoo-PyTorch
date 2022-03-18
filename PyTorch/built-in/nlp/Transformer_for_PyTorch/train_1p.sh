#! /bin/bash
DATA_DIR=./data/dataset/wmt14_en_de_joined_dict/
MODELDIR="./checkpoints/"
mkdir -p "$MODELDIR"
LOGFILE="$MODELDIR/log"
STAT_FILE="log.txt"

source ./npu_set_env.sh
source ./env_new.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export PTCOPY_ENABLE=1
export TASK_QUEUE_ENABLE=1
export DYNAMIC_OP="ADD#MUL"

python3 -u train_1p.py \
  ${DATA_DIR} \
  --device-id 7\
  --arch transformer_wmt_en_de \
  --share-all-embeddings \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.997 \
  --adam-eps "1e-9" \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 0.0 \
  --warmup-updates 4000 \
  --lr 0.0006 \
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-sentences 128\
  --max-tokens 102400\
  --max-epoch 40\
  --seed 1 \
  --save-dir $MODELDIR \
  --save-interval 1\
  --update-freq 8\
  --log-interval 1\
  --stat-file $STAT_FILE\
  --distributed-world-size 1\
  --amp\
  --amp-level O2