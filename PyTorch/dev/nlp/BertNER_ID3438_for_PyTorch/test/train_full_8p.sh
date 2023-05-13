#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
export RANK_SIZE=8
export JOB_ID=10087
RANK_ID_START=0
export HCCL_WHITELIST_DISABLE=1

# 数据集路径,保持为空,不需要修改
data_path=""

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="BertNER_ID3438_for_PyTorch"
#训练epoch
train_epochs=4.0
#训练batch_size
batch_size=24
TASK_NAME="cluener"

#维测参数，precision_mode需要模型审视修改
PREC=""
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
autotune=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		         if or not over detection, default is False
    --data_dump_flag	     data dump flag, default is False
    --data_dump_step		 data dump step, default is 10
    --profiling		         if or not profiling for performance debug, default is False
    --data_path		         source data of training
    -h/--help		         show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --precision_mode* ]];then
        apex_opt_level=`echo ${para#*=}`
        if [[ $apex_opt_level != "O0" ]] && [[ $apex_opt_level != "O1" ]] && [[ $apex_opt_level != "O2" ]] && [[ $apex_opt_level != "O3" ]]; then
            echo "[Error] para \"precision_mode\" must be config O0 or O1 or O2 or O3"
            exit 1
        fi
        PREC=$apex_opt_level
    
    elif [[ $para == --over_dump* ]];then
        over_dump=`echo ${para#*=}`
        over_dump_path=${cur_path}/output/overflow_dump
        mkdir -p ${over_dump_path}
    elif [[ $para == --data_dump_flag* ]];then
        data_dump_flag=`echo ${para#*=}`
        data_dump_path=${cur_path}/output/data_dump
        mkdir -p ${data_dump_path}
    elif [[ $para == --hf32 ]];then
        hf32=`echo ${para#*=}`
    elif [[ $para == --fp32 ]];then
        fp32=`echo ${para#*=}`
    elif [[ $para == --data_dump_step* ]];then
        data_dump_step=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
        profiling_dump_path=${cur_path}/output/profiling
        mkdir -p ${profiling_dump_path}
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

if [[ $PREC == "O0" ]];then
    prec=" "
else
    prec="--fp16 --fp16_opt_level "$PREC
fi

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#训练开始时间，不需要修改
start_time=$(date +%s)

#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/output/0 ];then
    rm -rf ${cur_path}/output/0
    mkdir -p ${cur_path}/output/0/ckpt
else
    mkdir -p ${cur_path}/output/0/ckpt
fi

DISTRIBUTED="-m torch.distributed.launch --nproc_per_node=${RANK_SIZE}"

nohup python3 ${DISTRIBUTED}  ${cur_path}/../run_ner_crf.py \
  --model_type=bert \
  --model_name_or_path=$data_path/prev_trained_model/bert-base-chinese \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  ${prec} \
  --do_lower_case \
  --data_dir=$data_path/datasets/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=${batch_size} \
  --per_gpu_eval_batch_size=24 \
  --learning_rate=2.4e-4 \
  --crf_learning_rate=8e-3 \
  --num_train_epochs=${train_epochs} \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=${cur_path}/output/0/ckpt \
  --overwrite_output_dir \
  --precision_mode=${precision_mode} \
  ${fp32} \
  ${hf32} \
  --seed=42 > ${cur_path}/output/0/train_0.log 2>&1 &

wait
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
sed -i 's/\x0D/\x0A/g' ${cur_path}/output/0/train_0.log
step_time=`grep "step cost" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "step cost" '{print$2}'|tail -n +4 | awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`
FPS=`awk 'BEGIN{printf "%.2f\n",'${RANK_SIZE}'*'${batch_size}'/'${step_time}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
train_accuracy=`grep -E "f1:.*loss:" $cur_path/output/0/train_0.log|awk 'END{print $15}'`
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
if [[ ${fp32} == "--fp32" ]];then
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'fp32'_'acc'
elif [[ ${hf32} == "--hf32" ]];then
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'hf32'_'acc'
else
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'
fi

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep -E "^\[Training\]" $cur_path/output/0/train_0.log|awk -F 'loss=' '{print$2}' | sed 's/[[:space:]]\].*//g' > $cur_path/output/0/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/0/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/0/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/0/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/0/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/0/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/0/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/0/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/0/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/0/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/0/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/0/${CaseName}.log
