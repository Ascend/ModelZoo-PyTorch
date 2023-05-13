#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
export NPU_CALCULATE_DEVICE=$ASCEND_DEVICE_ID
#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0

#进入到conda环境
#source activate py8


# 数据集路径,保持为空,不需要修改
data_path=""

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="FairSeq_Transformer_RT2_ID4042_for_PyTorch"
#训练epoch
train_epochs=2000
#训练batch_size
batch_size=32
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
    elif [[ $para == --hf32 ]];then
        hf32=`echo ${para#*=}`
    elif [[ $para == --fp32 ]];then
        fp32=`echo ${para#*=}`
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

if [[ $precision_mode == "must_keep_origin_dtype" ]];then
    PREC=""
else
    PREC=" --fp16 --fp16-scale-window 1500 "
fi

#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../

#python3 setup.py build_ext --inplace 
pip3 install --editable . 
#sed -i "s|pass|break|g" train.py
#sed -i "s|data/LibriSpeech|$data_path/LibriSpeech|g" config/libri/asr_example.yaml

#修改epoch参数


for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID
    
    
    
    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi
    
    #绑核，不需要绑核的模型删除，需要绑核的模型根据实际修改
    #cpucount=`lscpu | grep "CPU(s):" | head -n 1 | awk '{print $2}'`
    #cpustep=`expr $cpucount / 8`
    #echo "taskset c steps:" $cpustep
    #let a=RANK_ID*$cpustep
    #let b=RANK_ID+1
    #let c=b*$cpustep-1

    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    nohup python3 train.py $data_path/iwslt14.tokenized.de-en \
	    --arch transformer \
	    --optimizer adam \
	    --adam-betas '(0.9, 0.98)' \
	    --clip-norm 0.0 \
	    --lr 0.00006 \
	    --lr-scheduler inverse_sqrt \
	    --warmup-updates 4000 \
	    --device-id $ASCEND_DEVICE_ID \
	    --weight-decay 0.0001 \
	    --source-lang de \
	    --target-lang en \
	    --decoder-attention-heads 4 \
	    --decoder-ffn-embed-dim 1024 \
	    --encoder-attention-heads 4 \
	    --encoder-ffn-embed-dim 1024 \
	    --seed 12345 \
	    $PREC \
            --precision_mode $precision_mode \
            ${fp32} \
            ${hf32} \
	    --ddp-backend no_c10d \
	    --disable-validation \
	    --distributed-no-spawn \
	    --max-tokens 15000 \
	    --required-batch-size-multiple 32 \
        --max-epoch ${train_epochs} \
	    --max-source-positions 1024 \
	    --max-target-positions 1024 \
	    --num-workers 1 \
	    --log-interval 1 \
	    --save-interval 1 \
	    --share-decoder-input-output-embed > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done 
wait

#conda deactivate
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
#Time=`grep "iteration" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log |awk -F "time =" '{print $2}'|awk -F "ms" '{print $1}'|tail -n +6|awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`
#FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*1000/'${Time}'}'`
FPS=`grep -rn "wps=" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "wps=" '{print $2}' | awk -F "," '{print $1}' | awk '{if(NR>=325){print}}' | awk 'END {print}' |sed s/[[:space:]]//g`


#打印，不需要修改
echo "Final Performance images/sec : $FPS"

echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
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
#TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`
TrainingTime=`grep "iteration" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log |awk -F "time =" '{print $2}'|awk -F "ms" '{print $1}'| grep -v "^$" |tail -n +6|awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "loss=" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "loss=" '{print $2}'|awk -F "," '{print $1}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#输出训练精度,需要模型审视修改
train_accuracy=`tail -5 $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt|awk '{sum+=$1} END {print sum/NR}'`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log