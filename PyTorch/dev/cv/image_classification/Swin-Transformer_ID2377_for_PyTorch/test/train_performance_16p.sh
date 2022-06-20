#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`
export ASCEND_SLOG_PRINT_TO_STDOUT=0

#集合通信参数,不需要修改

export HCCL_WHITELIST_DISABLE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=23456

export RANK_SIZE=8
export JOB_ID=10087
RANK_ID_START=0













# 数据集路径,保持为空,不需要修改
data_path=""
conf_path=""
server_index=""
fix_node_ip=""
devicesnum=""


#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="Swin-Transformer_ID2377_for_PyTorch"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=32
#训练step
#train_steps=100
#学习率
learning_rate=0.99

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
PREC=""
# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1p.sh <args>"
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
        apex_opt_level=`echo ${para#*=}`
                    if [[ $apex_opt_level != "O1" ]] && [[ $apex_opt_level != "O2" ]] && [[ $apex_opt_level != "O3" ]]; then
                            echo "[ERROR] para \"precision_mode\" must be config O1 or O2 or O3"
                            exit 1
                    fi
        PREC="--apex --apex-opt-level "$apex_opt_level

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
		elif [[ $para == --fix_node_ip* ]];then
	    fix_node_ip=`echo ${para#*=}`
	elif [[ $para == --devicesnum* ]];then
	    devicesnum=`echo ${para#*=}`
    elif [[ $para == --conf_path* ]];then
            conf_path=`echo ${para#*=}`
    elif [[ $para == --server_index* ]];then
            server_index=`echo ${para#*=}`

    fi
done

one_node_ip=`find $conf_path -name "server_*0.info"|awk -F "server_" '{print $2}'|awk -F "_" '{print $1}'`
linux_num=`find $conf_path -name "server_*.info" |wc -l`

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../


#sed -i "s|../datasets|$data_path|g" pytorch-1-ann-simple-cnn.py
sed -i "s|TRAIN.EPOCHS = 300|TRAIN.EPOCHS = 1|g" config.py
sed -i "s|pass|break|g" main.py

#mkdir -p checkpoints
#mkdir -p /root/.cache/torch/hub/checkpoints
#cp $data_path/fcn_* /root/.cache/torch/hub/checkpoints

export HCCL_IF_IP=$fix_node_ip
export MASTER_ADDR=$one_node_ip
export MASTER_PORT=29688
export HCCL_WHITELIST_DISABLE=1
device_num=${#devicesnum}
devices_num=`awk 'BEGIN{printf "%.0f\n",'${device_num}'-1}'`
NPUS=($(seq 0 $devices_num))
rank_server=`awk 'BEGIN{printf "%.0f\n",'${device_num}'*'${server_index}'}'`
export WORLD_SIZE=`awk 'BEGIN{printf "%.0f\n",'${device_num}'*'${linux_num}'}'`

rank=0
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID
   export RANK=$RANK_ID 
    
   export ASCEND_DEVICE_ID=$RANK_ID    
    export NPU_CALCULATE_DEVICE=$RANK_ID    
     export RANK=`awk 'BEGIN{printf "%.0f\n",'${rank}'+'${rank_server}'}'`  
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
    nohup python3 main.py --cfg configs/swin_t_64.yaml $PREC --local_rank $RANK_ID --data-path $data_path --batch-size 32  > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

	let rank++
done 
wait

#恢复参数
#sed -i "s|$data_path|../datasets|g" pytorch-1-ann-simple-cnn.py
sed -i "s|TRAIN.EPOCHS = 1|TRAIN.EPOCHS = 300|g" config.py
sed -i "s|break|pass|g" main.py

#conda deactivate
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep FPS  $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "FPS:" '{print $2}'|tail -n +2|awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`
#FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*'${perf}'}'`
FPS=`awk 'BEGIN{printf "%.0f\n",'${FPS}'*'16'}'`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep "Max accuracy" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk -F " " '{print $9}'|awk 'NR==1{max=$1;next}{max=max>$1?max:$1}END{print max}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"
RANK_SIZE_NEW=`awk 'BEGIN{printf "%.0f\n",'${RANK_SIZE}'*'${linux_num}'}'`
#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE_NEW}'p'_'perf'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "INFO Train" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "loss " '{print $2}'|awk -F " " '{print $1}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE_NEW}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
