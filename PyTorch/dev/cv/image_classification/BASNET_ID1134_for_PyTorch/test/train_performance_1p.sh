#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`/../

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
Network="BASNET_ID1134_for_PyTorch"
#训练epoch
train_epochs=1
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
        over_dump_path=${cur_path}/test/output/overflow_dump
        mkdir -p ${over_dump_path}
    elif [[ $para == --data_dump_flag* ]];then
        data_dump_flag=`echo ${para#*=}`
        data_dump_path=${cur_path}/test/output/data_dump
        mkdir -p ${data_dump_path}
    elif [[ $para == --data_dump_step* ]];then
        data_dump_step=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
        profiling_dump_path=${cur_path}/test/output/profiling
        mkdir -p ${profiling_dump_path}
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --conda_name* ]];then
        conda_name=`echo ${para#*=}`
	    source set_conda.sh
	    source activate $conda_name
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
cd $cur_path

sed -i "s|data_dir = './train_data/'|data_dir = '$data_path/'|g" basnet_train.py
sed -i "s|tra_image_dir = 'DUTS/DUTS-TR/DUTS-TR/im_aug/'|tra_image_dir = 'DUTS-TR/DUTS-TR-Image/'|g" basnet_train.py
sed -i "s|tra_label_dir = 'DUTS/DUTS-TR/DUTS-TR/gt_aug/'|tra_label_dir = 'DUTS-TR/DUTS-TR-Mask/'|g" basnet_train.py
sed -i "s|epoch_num = 100000|epoch_num = $train_epochs|g" basnet_train.py
sed -i "s|batch_size_train = 8 |batch_size_train = $batch_size|g" basnet_train.py
sed -i "s|lr=0.001|lr=$learning_rate|g" basnet_train.py


mkdir -p $cur_path/saved_models/basnet_bsi

mkdir -p /root/.cache/torch/checkpoints/
cp -r $data_path/resnet34-333f7ec4.pth /root/.cache/torch/checkpoints/

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID



    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
    fi
    # 绑核，不需要的绑核的模型删除，需要的模型审视修改
    #let a=RANK_ID*12
    #let b=RANK_ID+1
    #let c=b*12-1

    
    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path
    nohup python3 basnet_train.py \
        $PREC \
        --apex \
        --device_id=$ASCEND_DEVICE_ID > ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done 
wait
sed -i "s|epoch_num = $train_epochs|epoch_num = 100000|g" basnet_train.py
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep "fps:" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|tail -n +3 | awk -F ":" '{print$NF}' |awk '{sum+=$1} END {print"",sum/NR}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
#输出编译时间
CompileTime=`grep train_time:  $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log| head -2 |awk -F "train_time: " '{print $2}' | awk '{sum+=$1} END {print"",sum}' |sed s/[[:space:]]//g`

#输出训练精度,需要模型审视修改
train_accuracy=`grep "acc" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F ":" 'END {print $5}'|cut -c 1-6`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
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
#TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`
TrainingTime=`awk -v x="$FPS" -v y=8 'BEGIN{printf "%.3f\n",y*1000/x}'`


#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep epoch $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| awk '{print$10}' | sed 's/,//g' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CompileTime = ${CompileTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log