#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`
#source ../env_npu.sh

data_path=""
conf_path=""
server_index=""
fix_node_ip=""
#集合通信参数,不需要修改

export RANK_SIZE=16

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="BertBase_ID0490_for_PyTorch"
#训练batch_size
batch_size=80


#参数校验，不需要修改
for para in $*
do
    if [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --over_dump* ]];then
        over_dump=`echo ${para#*=}`
        over_dump_path=${cur_path}/output/overflow_dump
        mkdir -p ${over_dump_path}
    elif [[ $para == --data_dump_flag* ]];then
        data_dump_flag=`echo ${para#*=}`
        data_dump_path=${cur_path}/output/data_dump
        mkdir -p ${data_dump_path}
    elif [[ $para == --data_dump_step* ]];then
        data_dump_step=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
        profiling_dump_path=${cur_path}/output/profiling
        mkdir -p ${profiling_dump_path}
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
    elif [[ $para == --conf_path* ]];then
            conf_path=`echo ${para#*=}`
    elif [[ $para == --server_index* ]];then
            server_index=`echo ${para#*=}`
    elif [[ $para == --fix_node_ip* ]];then
            fix_node_ip=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

one_node_ip=`find $conf_path -name "server_*0.info"|awk -F "server_" '{print $2}'|awk -F "_" '{print $1}'`
linux_num=`find $conf_path -name "server_*.info" |wc -l`

export HCCL_IF_IP=$fix_node_ip
export MASTER_ADDR=$one_node_ip

rank_server=`awk 'BEGIN{printf "%.0f\n",8*'${server_index}'}'`
export NPU_WORLD_SIZE=`awk 'BEGIN{printf "%.0f\n",8*'${linux_num}'}'`

#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../
cur_1=${1:-"1"}
cur_2=${2:-"2"}
cur_3=${3:-"3"}
cur_4=${4:-"4"}
init_checkpoint=${5:-"`${data_path}/pretrained/bert_base_pretrain.pt`"}
epochs=${6:-"1.0"}
batch_size=${7:-"80"}
learning_rate=${8:-"2e-4"}
precision=${9:-"fp16"}
num_npu=${10:-"16"}
seed=${11:-"1"}
squad_dir=${12:-"`${data_path}/squad/v1.1`"}
vocab_file=${13:-"data/uncased_L-24_H-1024_A-16/vocab.txt"}
OUT_DIR=${14:-"results/SQuAD"}
mode=${15:-"train eval"}
CONFIG_FILE=${16:-"bert_base_config.json"}
max_steps=${17:-"-1"}

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
CMD+="--init_checkpoint=${data_path}/pretrained/bert_base_pretrain.pt "
if [ "$mode" = "train" ] ; then
  CMD+="--do_train "
  CMD+="--train_file=${data_path}/squad/v1.1/train-v1.1.json "
  CMD+="--train_batch_size=$batch_size "
elif [ "$mode" = "eval" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=${data_path}/squad/v1.1/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=${data_path}/squad/v1.1/evaluate-v1.1.py "
  CMD+="--do_eval "
elif [ "$mode" = "prediction" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=${data_path}/squad/v1.1/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
else
  CMD+=" --do_train "
  CMD+=" --train_file=${data_path}/squad/v1.1/train-v1.1.json "
  CMD+=" --train_batch_size=$batch_size "
  CMD+="--do_predict "
  CMD+="--predict_file=${data_path}/squad/v1.1/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=${data_path}/squad/v1.1/evaluate-v1.1.py "
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
CMD+=" --addr=$one_node_ip"

if [ $(uname -m) = "aarch64" ]
then
  for i in $(seq 0 7)
  do
  let p_start=0+24*i
  let p_end=23+24*i
  export RANK=`awk 'BEGIN{printf "%.0f\n",'${i}'+'${rank_server}'}'`
  if [ -d ${cur_path}/output/${i} ];then
        rm -rf ${cur_path}/output/${i}
        mkdir -p ${cur_path}/output/$i
  else
        mkdir -p ${cur_path}/output/$i
  fi
  taskset -c $p_start-$p_end $CMD --local_rank=$i > ${cur_path}/output/${i}/train_${i}.log 2>&1 &
  done
else
  for i in $(seq 0 7)
  do
  if [ -d ${cur_path}/output/${i} ];then
        rm -rf ${cur_path}/output/${i}
        mkdir -p ${cur_path}/output/$i
  else
        mkdir -p ${cur_path}/output/$i
  fi
  $CMD --local_rank=$i > ${cur_path}/output/${i}/train_${i}.log 2>&1 &
  done
fi
wait

ASCEND_DEVICE_ID=0
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
iter=`grep 'Epoch: ' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "iter/s :" '{print $NF}'|awk 'NR==1{max=$1;next}{max=max>$1?max:$1}END{print max}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${iter}'*16*'${batch_size}'}'`
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

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep -r "step_loss :" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print $19}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
rm -rf ${data_path}/squad/v1.1/train-v1.1.json_bert-large-uncased_384_128_64