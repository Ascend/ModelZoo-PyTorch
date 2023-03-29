#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0


# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="SINGLESHOTPOSE_ID0869_for_PyTorch"
#训练epoch
train_epochs=2
#训练batch_size
batch_size=8
#训练step
train_steps=
#学习率
learning_rate=0.001

#TF2.X独有，需要模型审视修改
#export NPU_LOOP_SIZE=${train_steps}

#维测参数，precision_mode需要模型审视修改
#precision_mode="allow_mix_precision"
PREC=""
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1P.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(O0/O1/O2/O3)
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
    elif [[ $para == --conda_name* ]];then
        conda_name=`echo ${para#*=}`
	        source set_conda.sh
	        source activate $conda_name
    elif [[ $para == --ci_cp* ]];then
        ci_cp=`echo ${para#*=}`
    fi
done

if [[ $ci_cp == "1" ]];then
    cp -r $data_path ${data_path}_bak
fi


#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../
sed -i "s|train  = LINEMOD/iron/train.txt|train  = $data_path/LINEMOD/iron/train.txt|g" cfg/iron.data
sed -i "s|valid  = LINEMOD/iron/test.txt|valid  = $data_path/LINEMOD/iron/test.txt|g" cfg/iron.data
sed -i "s|backup = backup/iron|backup = $data_path/backup/iron|g" cfg/iron.data
sed -i "s|mesh = LINEMOD/iron/iron.ply|mesh = $data_path/LINEMOD/iron/iron.ply|g" cfg/iron.data
sed -i "s|tr_range = LINEMOD/iron/training_range.txt|tr_range = $data_path/LINEMOD/iron/training_range.txt|g" cfg/iron.data\
sed -i "s|batch=8|batch=$batch_size|g" cfg/yolo-pose.cfg
sed -i "s|max_epochs=500|max_epochs=$train_epochs|g" cfg/yolo-pose.cfg
sed -i "s|learning_rate=0.001|learning_rate=$learning_rate|g" cfg/yolo-pose.cfg

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
    # 绑核，不需要的绑核的模型删除，需要的模型审视修改
    #let a=RANK_ID*12
    #let b=RANK_ID+1
    #let c=b*12-1

    sed -i "s|'VOCdevkit/VOC2012/JPEGImages'|'$data_path/VOCdevkit/VOC2012/JPEGImages'|g" train.py
    sed -i "s|data/LINEMOD/iron/JPEGImages|$data_path/LINEMOD/iron/JPEGImages|g" $data_path/LINEMOD/iron/train.txt
    sed -i "s|data/LINEMOD/iron/JPEGImages|$data_path/LINEMOD/iron/JPEGImages|g" $data_path/LINEMOD/iron/test.txt
    
    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path
    nohup python3 train.py \
        --datacfg cfg/iron.data \
        --modelcfg cfg/yolo-pose.cfg \
        $PREC \
        --device_id=$ASCEND_DEVICE_ID \
        --initweightfile $data_path/darknet19_448.conv.23 \
        --pretrain_num_epochs 2 > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done 
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep "fps" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $NF}'| tail -n -10 |awk '{sum+=$1} END {print sum/NR}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
#输出编译时间
CompileTime=`grep "step_time" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log| head -2 |awk -F "step_time: " '{print $2}' | awk '{sum+=$1} END {print"",sum}' |sed s/[[:space:]]//g`

#输出训练精度,需要模型审视修改
#train_accuracy=`grep "acc" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F ":" 'END {print $5}'|cut -c 2-9`
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"
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
grep "total" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $NF}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
echo "CompileTime = ${CompileTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
if [[ $ci_cp == "1" ]];then
    rm -rf $data_path
    mv ${data_path}_bak $data_path
fi