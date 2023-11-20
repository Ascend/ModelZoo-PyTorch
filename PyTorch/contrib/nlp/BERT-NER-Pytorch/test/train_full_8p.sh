#!/bin/bash
################基础配置参数，需要模型审视修改##################

CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base-chinese
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="cluener"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12355
export TASK_QUEUE_ENABLE=2
export ASCEND_DEVICE_ID=0
###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi
start_time=$(date +%s) 
python -m torch.distributed.launch --nproc_per_node 8 \
  --master_addr $MASTER_ADDR --master_port $MASTER_PORT run_ner_crf.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=24 \
  --per_gpu_eval_batch_size=24 \
  --learning_rate=3e-5 \
  --crf_learning_rate=1e-3 \
  --num_train_epochs=4.0 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42 > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train.log 2>&1
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

echo "------------------ Final result ------------------"
lines=$(grep -P "\[Training\] \d+/\d+.*s/step" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train.log)
nlines=10
use_lines=$(echo "${lines}" | tail -n ${nlines})
perfs=$(echo "${use_lines}" | sed -E 's/.*\[Training\].*\[=+\] ([0-9]+\.[0-9]+)([m]*)s.*/\1/')
sum=$(echo "${perfs}" | tr '\n' '+' | sed -E 's/(.*)\+$/\1/')
sum=$(echo "${sum}" | bc -l)
avg=$(echo "${sum} / 10" | bc -l)
echo "Final performance WPS: $avg"
line=$(cat ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train.log | grep -P "f1:" | head -n 1)
acc=$(echo "${line}" | grep -P ".*acc.*recall.*f1.*" | sed -E 's/.*f1: ([0-9]+\.[0-9]+).*/\1/')
echo "Final Best Acc (F1 score): $acc"
echo "E2E training duration sec: $e2e_time"



  




