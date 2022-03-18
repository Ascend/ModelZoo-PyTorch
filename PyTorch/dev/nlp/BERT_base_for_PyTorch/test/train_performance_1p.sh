#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`
source ../env_npu.sh

#集合通信参数,不需要修改

export RANK_SIZE=1

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="Bert-base_for_PyTorch"
#训练epoch
train_epochs=2
#训练batch_size
batch_size=80


#维测参数，precision_mode需要模型审视修改
precision_mode="allow_fp32_to_fp16"


#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../
init_checkpoint=${1:-"checkpoints/bert_base_pretrain.pt"}
epochs=${2:-"1.0"}
batch_size=${3:-"80"}
learning_rate=${4:-"8e-5"}
precision=${5:-"fp16"}
seed=${6:-"1"}
squad_dir=${7:-"data/squad/v1.1"}
vocab_file=${8:-"data/uncased_L-24_H-1024_A-16/vocab.txt"}
OUT_DIR=${9:-"results/SQuAD"}
mode=${10:-"train eval"}
CONFIG_FILE=${11:-"bert_base_config.json"}
max_steps=${12:-"-1"}
npu_id=${13:-"0"}

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

if [ $(uname -m) = "aarch64" ]; then
	CMD="python3.7 run_squad.py "
else
    CMD="python3.7 run_squad.py "
fi
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
CMD+=" --npu_id=$npu_id "
CMD+=" --loss_scale=4096 "

if [ -d ${cur_path}/output/${npu_id} ];then
      rm -rf ${cur_path}/output/${npu_id}
      mkdir -p ${cur_path}/output/$npu_id
else
      mkdir -p ${cur_path}/output/$npu_id
fi
$CMD> ${cur_path}/output/${npu_id}/train_${npu_id}.log 2>&1 &
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
iter=`grep 'Epoch:' $cur_path/output/${npu_id}/train_${npu_id}.log|awk -F "iter/s :" '{print $NF}'|awk 'NR==1{max=$1;next}{max=max>$1?max:$1}END{print max}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${iter}'*'${batch_size}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`

#从train_$npu_id.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep -r "step_loss :" $cur_path/output/$npu_id/train_$npu_id.log | awk '{print $19}' > $cur_path/output/$npu_id/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$npu_id/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$npu_id/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$npu_id/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$npu_id/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$npu_id/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$npu_id/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$npu_id/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$npu_id/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$npu_id/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$npu_id/${CaseName}.log