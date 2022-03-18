#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export NPU_CALCULATE_DEVICE=$ASCEND_DEVICE_ID

#集合通信参数,不需要修改
export RANK_SIZE=8
export JOB_ID=10087
RANK_ID_START=0


#export HCCL_WHITELIST_DISABLE=1
#export MASTER_ADDR=127.0.0.1
#export MASTER_PORT=23456
#export RANK=0
#export WORLD_SIZE=$NNPU
RANK_ID=0
export NNPU=8
export WORLD_SIZE=$NNPU
export MASTER_ADDR=$(hostname -I |awk '{print $1}')
export MASTER_PORT=23456




# 数据集路径,保持为空,不需要修改
data_path=""

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="XLM_ID0740_for_PyTorch"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=16
#训练step
#train_steps=`expr 1281167 / ${batch_size}`
#学习率
learning_rate=0.495

#TF2.X独有，不需要修改
#export NPU_LOOP_SIZE=${train_steps}

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
autotune=False

# 帮助信息，不h需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump                  if or not over detection, default is False
    --data_dump_flag         data dump flag, default is False
    --data_dump_step             data dump step, default is 10
    --profiling                  if or not profiling for performance debug, default is False
    --data_path                  source data of training
    -h/--help                    show help message
    "
    exit 1
fi


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
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../


#sed -i "s|./data|$data_path|g" examples/cats_and_dogs.py
#sed -i "s|epochs = 20|epochs = 1|g" examples/cats_and_dogs.py
sed -i "s|pass|break|g" train.py

#python3 setup.py install
#mkdir -p checkpoints
#mkdir -p /root/.cache/torch/hub/checkpoints
#cp $data_path/fcn_* /root/.cache/torch/hub/checkpoints


#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
else
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
fi



#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
export MASTER_ADDR=localhost
export MASTER_PORT=29688
export HCCL_WHITELIST_DISABLE=1

NPUS=($(seq 0 7))
export NPU_WORLD_SIZE=${#NPUS[@]}
rank=0

KERNEL_NUM=$(($(nproc)/8))
#for((RANK_ID=0;RANK_ID<NNPU;RANK_ID++))

for i in ${NPUS[@]}
do  
    mkdir -p  $cur_path/output/${i}/
    export NPU_CALCULATE_DEVICE=${i}
    export RANK=${rank}
    export ASCEND_DEVICE_ID=${i}
    echo run process ${rank} 
    if [ $(uname -m) = "aarch64" ]
    then
        PID_START=$((KERNEL_NUM * i))
        PID_END=$((PID_START + KERNEL_NUM - 1))
        taskset -c $PID_START-$PID_END python3 train.py --exp_name xlm_en_zh \
            --dump_path ./dumped        \
            --data_path $data_path/50k      \
            --lgs 'en-zh'          \
            --clm_steps ''          \
            --mlm_steps 'en,zh'          \
            --emb_dim 1024               \
            --n_layers 12                \
            --n_heads 16                 \
            --dropout 0.1                \
            --attention_dropout 0.1      \
            --gelu_activation true       \
            --batch_size $batch_size             \
            --bptt 256                   \
            --optimizer npu_fused_adam_v2,lr=0.00005     \
            --epoch_size 300000               \
            --max_epoch $train_epochs                    \
            --validation_metrics _valid_mlm_ppl          \
            --stopping_criterion _valid_mlm_ppl,8       \
            --fp16 true     \
            --amp 2 \
            --seed 1 \
            --local_rank $rank > $cur_path/output/$ASCEND_DEVICE_ID/train_${i}.log 2>&1 &
    let rank++
    else
        python3 train.py --exp_name xlm_en_zh \
            --dump_path ./dumped        \
            --data_path $data_path/50k      \
            --lgs 'en-zh'          \
            --clm_steps ''          \
            --mlm_steps 'en,zh'          \
            --emb_dim 1024               \
            --n_layers 12                \
            --n_heads 16                 \
            --dropout 0.1                \
            --attention_dropout 0.1      \
            --gelu_activation true       \
            --batch_size $batch_size              \
            --bptt 256                   \
            --optimizer npu_fused_adam_v2,lr=0.00005     \
            --epoch_size 300000               \
            --max_epoch $train_epochs                     \
            --validation_metrics _valid_mlm_ppl          \
            --stopping_criterion _valid_mlm_ppl,8       \
            --fp16 true     \
            --amp 2 \
            --seed 1 \
            --local_rank $rank > $cur_path/output/$ASCEND_DEVICE_ID/train_${i}.log 2>&1 &
    let rank++
    fi
done

wait



#恢复参数
sed -i "s|break|pass|g" train.py

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep "sent/s"  $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "sent/s -" '{print $2}'|awk '{print $1}'|tail -n +2|awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`
FPS=`awk 'BEGIN{printf "%.2f\n",'${FPS}'*'8'}'`

#FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*'${perf}'}'`


#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
#train_accuracy=`grep eval_accuracy $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|grep -v mlp_log|awk 'END {print $5}'| sed 's/,//g' |cut -c 1-5`
#train_accuracy=`grep "vaild_en_mlm_acc" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "vaild_en_mlm_acc ->" '{print $2}'|awk 'NR==1{max=$1;next}{max=max>$1?max:$1}END{print max}'`
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"
#echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "sent/s" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "MLM-en:  " '{print $2}'|awk '{print $1}' >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
#echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log