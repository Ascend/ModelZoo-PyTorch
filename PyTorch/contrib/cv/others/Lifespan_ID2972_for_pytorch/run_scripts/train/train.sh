#!/bin/bash
#Usage: ./train.sh data_url train_url sex
CUR_DIR=$( cd `dirname $0` ; pwd )
SCRIPT_DIR=$( cd `dirname $CUR_DIR` ; pwd )
ROOT_DIR=$( cd `dirname $SCRIPT_DIR` ; pwd )
DATA_URL=$1
TRAIN_URL=$2
MODEL_SEX=$3
BATCH_SIZE=3
DATA_ROOT=$DATA_URL$MODEL_SEX
CHECKPOINTS_DIR="$TRAIN_URL"checkoutpoints

export ASCEND_GLOBAL_LOG_LEVEL=3

if [ ! -d $CHECKPOINTS_DIR ];then
  mkdir $CHECKPOINTS_DIR
fi
function print_inputvar() {
  echo "MODEL_SEX      :$MODEL_SEX"
  echo "BATCH_SIZE     :$BATCH_SIZE"
  echo "DATAROOT       :$DATA_ROOT"
  echo "CHECKPOINTS_DIR:$CHECKPOINTS_DIR"
}
echo "-----------------Current Variable-----------------"
print_inputvar
# solove model name.
MODEL_BAK="_model"
MODEL_NAME=$MODEL_SEX$MODEL_BAK
# cmd for train.
cd $ROOT_DIR
TRAIN_CMD="python3 -u train.py --npu_ids 0 --dataroot $DATA_ROOT --name $MODEL_NAME --batchSize $BATCH_SIZE --verbose --display_id 0 --checkpoints_dir $CHECKPOINTS_DIR --amp"
echo "Will run    :$TRAIN_CMD"
$TRAIN_CMD
#python3 -u train.py --npu_ids 0 --dataroot /home/ma-user/modelarts/inputs/data_url_0/males --name males_model --batchSize 3 --verbose --display_id 0