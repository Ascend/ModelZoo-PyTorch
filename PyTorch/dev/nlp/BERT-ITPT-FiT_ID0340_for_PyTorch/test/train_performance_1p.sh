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
#export ASCEND_GLOBAL_LOG_LEVEL=3
#export ASCEND_SLOG_PRINT_TO_STDOUT=0

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="BERT-ITPT-FiT_ID0340_for_PyTorch"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=32
#训练step
train_steps=100
#学习率
learning_rate=8e-2
RANK_SIZE=1

#维测参数，precision_mode需要模型审视修改
#precision_mode="allow_mix_precision"
# PREC="--apex --apex-opt-level O2"
#维持参数，以下不需要修改
#over_dump=False
#data_dump_flag=False
#data_dump_step="10"
#profiling=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --data_path		           source data of training
    --train_steps                  train steps
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
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --train_steps* ]];then
        train_steps=`echo ${para#*=}`
    fi
done


#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#进入训练脚本目录，需要模型审视修改
cd $cur_path
#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];then
   rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
   mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
else
   mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
fi
    
sed -i "s|data/bert-base-cased/|$data_path/bert-base-cased|g" main.py
sed -i "s|data/aclImdb/train|$data_path/aclImdb/train|g" main.py
sed -i "s|data/aclImdb/test|$data_path/aclImdb/test|g" main.py
sed -i "s|NUM_EPOCHS = 10|NUM_EPOCHS = 1|g" main.py

#训练开始时间，不需要修改
start_time=$(date +%s)
ln -s ${data_path}
#find /usr -name site-packages | xargs -I {} mv {}/apex {}/apex_bak
nohup python3 main.py  > $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait
#训练结束时间，不需要修改
#find /usr -name site-packages | xargs -I {} mv {}/apex_bak {}/apex
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

sed -i "s|$data_path/bert-base-cased/|data/bert-base-cased|g" main.py
sed -i "s|$data_path/aclImdb/train|data/aclImdb/train|g" main.py
sed -i "s|$data_path/aclImdb/test|data/aclImdb/test|g" main.py
sed -i "s|NUM_EPOCHS = 1|NUM_EPOCHS = 10|g" main.py
#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
#计算单步时间
TrainingTime=`grep "Train" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk -F "Train-Time: " '{print $2}' | tail -n+3 | awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`

#计算FPS

FPS=`python3 -c "print(${batch_size}/${TrainingTime})"`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep "Train" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F "Train-Accuracy: " '{print $2}'|awk -F " | Train-Time" '{print $1}' |awk 'BEGIN {max = 0} {if ($1+0 > max+0) max=$1} END {print max}'`
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
#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "Train" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F "Train-Loss: " '{print $2}'|awk -F " | " '{print $1}' >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`grep Train-Loss $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log| tail -1 | awk '{print $2}'`
#`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`


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
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log

###下面字段用于冒烟看护
#BatchSize=${batch_size}
#设备类型，自动获取
#DeviceType=`uname -m`
#用例名称，自动获取
#CaseName=${Network}_bs${BatchSize}_${RankSize}'p'_'perf'

##获取错误信息
#系统错误信息
#error_msg="RuntimeError: derivative for npu_layer_norm_eval is not implemented"
#判断错误信息是否和历史状态一致，此处无需修改
#Status=`grep "${error_msg}" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | wc -l`
#失败阶段，枚举值图准备FAIL/图拆分FAIL/图优化FAIL/图编译FAIL/图执行FAIL/流程OK
#ModelStatus="Parser FAIL"
#DTS单号或者issue链接
#DTS_Number="https://gitee.com/ascend/pytorch/issues/I3TWK4?from=project-issue"

#关键信息打印到CaseName.log中，此处无需修改
#echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "RankSize = ${RankSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "ModelStatus = ${ModelStatus}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "DTS_Number = ${DTS_Number}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "Status = ${Status}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "error_msg = ${error_msg}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
