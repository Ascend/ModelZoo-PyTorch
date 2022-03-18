#!/usr/bin/env bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
source env_npu.sh

init_checkpoint=${1:-"checkpoints/bert_base_pretrained_amp.pt"}
epochs=${2:-"2.0"}
batch_size=${3:-"96"}
learning_rate=${4:-"2e-4"}
precision=${5:-"fp16"}
num_npu=${6:-"8"}
seed=${7:-"1"}
squad_dir=${8:-"data/squad/v1.1"}
vocab_file=${9:-"data/uncased_L-24_H-1024_A-16/vocab.txt"}
OUT_DIR=${10:-"results/SQuAD"}
mode=${11:-"train eval"}
CONFIG_FILE=${12:-"bert_base_config.json"}
max_steps=${13:-"-1"}
RANK_SIZE=$num_npu

echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16=" --fp16 "
fi

CMD="python3.7 run_squad.py "
CMD+="--init_checkpoint=$init_checkpoint "
if [ "$mode" = "train" ] ; then
  CMD+="--do_train "
  CMD+="--train_file=$squad_dir/train-v1.1.json "
  CMD+="--train_batch_size=$batch_size "
elif [ "$mode" = "eval" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do_eval "
elif [ "$mode" = "prediction" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
else
  CMD+=" --do_train "
  CMD+=" --train_file=$squad_dir/train-v1.1.json "
  CMD+=" --train_batch_size=$batch_size "
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do_eval "
fi

CMD+=" --do_lower_case "
CMD+=" --bert_model=bert-large-uncased "
CMD+=" --learning_rate=$learning_rate "
CMD+=" --seed=$seed "
CMD+=" --num_train_epochs=$epochs "
CMD+=" --max_seq_length=384 "
CMD+=" --doc_stride=128 "
CMD+=" --output_dir=$OUT_DIR "
CMD+=" --vocab_file=$vocab_file "
CMD+=" --config_file=$CONFIG_FILE "
CMD+=" --max_steps=$max_steps "
CMD+=" $use_fp16"
CMD+=" --use_npu"
CMD+=" --num_npu=$num_npu"
CMD+=" --loss_scale=4096"
CMD+=" --addr=127.0.0.1"

if [ $(uname -m) = "aarch64" ]
then
  for i in $(seq 0 7)
  do
    corenum=`cat /proc/cpuinfo |grep "processor"|wc -l`
    let a=$i*${corenum}/${RANK_SIZE}
    let b=$i+1
    let c=b*${corenum}/${RANK_SIZE}-1
    taskset -c $a-$c  $CMD --local_rank=$i &
  done 
else
  for i in $(seq 0 7)
  do
    $CMD --local_rank=$i &
  done
fi
wait
