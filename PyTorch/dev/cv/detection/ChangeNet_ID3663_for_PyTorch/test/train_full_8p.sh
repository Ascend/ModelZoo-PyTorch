#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
export RANK_SIZE=8
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="ChangeNet_ID3663_for_PyTorch"
#训练epoch
train_epochs=50
#训练batch_size
batch_size=20

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
        if [[ $apex_opt_level != "O1" ]] && [[ $apex_opt_level != "O2" ]] && [[ $apex_opt_level != "O3" ]]; then
            echo "[Error] para \"precision_mode\" must be config O1 or O2 or O3"
            exit 1
        fi
        PREC="--amp --amp_opt_level "$apex_opt_level
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
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

export WORLD_SIZE=8
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="12345"

#训练开始时间，不需要修改
start_time=$(date +%s)
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
  export RANK=$RANK_ID
  export LOCAL_RANK=$RANK_ID
  #创建DeviceID输出目录，不需要修改
  if [ -d ${cur_path}/output/${LOCAL_RANK} ];then
      rm -rf ${cur_path}/output/${LOCAL_RANK}/ckpt
      mkdir -p ${cur_path}/output/${LOCAL_RANK}/ckpt
  else
      mkdir -p ${cur_path}/output/${LOCAL_RANK}/ckpt
  fi

  nohup python3 ${cur_path}/../Train.py \
    --num_epochs=${train_epochs} \
    --batch_size=${batch_size} \
    --num_classes=11 \
    --local_rank=${LOCAL_RANK} \
    --data_dir=${data_path} \
    --output_dir=${cur_path}/output/${LOCAL_RANK}/ckpt \
    --fp16 > ${cur_path}/output/${LOCAL_RANK}/train_${LOCAL_RANK}.log 2>&1 &
done
wait

#conda deactivate
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
fps=`grep "FPS" $cur_path/output/0/train_0.log | awk '{print $3}'| awk '{sum+=$1} END {print sum/NR}' | sed 's/[[:space:]]//g'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${RANK_SIZE}'*'${fps}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
train_accuracy=`grep "Best val Acc:" $cur_path/output/0/train_0.log | awk '{print$4}'`
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${FPS}'}'`

#从train_0.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "train Loss:" $cur_path/output/0/train_0.log|awk '{print$3}' > $cur_path/output/0/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/0/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >> $cur_path/output/0/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/0/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/0/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/0/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/0/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/0/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/0/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/0/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/0/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/0/${CaseName}.log
