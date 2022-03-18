#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
RANK_ID_START=0
export RANK_SIZE=1
data_path=""
#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="Bert-base_for_PyTorch"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=80
learning_rate=8e-5

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1P.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		           if or not over detection, default is False
    --data_dump_flag		     data dump flag, default is False
    --data_dump_step		     data dump step, default is 10
    --profiling		           if or not profiling for performance debug, default is False
    --data_path		           source data of training
    -h/--help		             show help message
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
    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
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

    
    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path
    nohup python3.7 run_squad.py \
          --init_checkpoint ${data_path}/pretrained/bert_base_pretrain.pt \
          --bert_model bert-large-uncased \
		  --do_train \
		  --train_file ${data_path}/squad/v1.1/train-v1.1.json \
		  --train_batch_size ${batch_size} \
		  --do_predict \
		  --predict_batch_size ${batch_size} \
		  --predict_file ${data_path}/squad/v1.1/dev-v1.1.json \
		  --learning_rate ${learning_rate} \
		  --num_train_epochs ${train_epochs} \
		  --seed 1 \
		  --fp16 \
		  --max_steps 100 \
		  --use_npu \
		  --loss_scale 4096 \
		  --vocab_file "data/uncased_L-24_H-1024_A-16/vocab.txt" \
		  --do_eval \
          --eval_script ${data_path}/squad/v1.1/evaluate-v1.1.py \
		  --npu_id ${ASCEND_DEVICE_ID} \
		  --do_lower_case \
		  --output_dir ${cur_path}/../results \
		  --config_file bert_base_config.json \
          --json-summary ${cur_path}/output/${ASCEND_DEVICE_ID}/dllogger.json > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done 
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