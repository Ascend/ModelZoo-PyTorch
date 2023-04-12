#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="Bertbase-crf"
# 训练batch_size
export BATCH_SIZE=192
# 训练使用的npu卡数
export WORLD_SIZE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
# 数据集路径,保持为空,不需要修改
data_path=""

# 训练epoch 20
train_epochs=20
# 加载数据进程数
workers=24
# 学习率
lr=3.2e-4
# 混合精度模式
opt_level="O2"
# warmup factor
warm_factor=0.4

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#################创建日志输出目录，不需要修改#################
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

KERNEL_NUM=$(($(nproc)/8))
export OMP_NUM_THREADS=$KERNEL_NUM
for i in $(seq 0 7)
do
if [ -d ${cur_path}/test/output/${i} ];
then
        rm -rf ${cur_path}/test/output/${i}
        mkdir -p ${cur_path}/test/output/${i}
else
        mkdir -p ${cur_path}/test/output/${i}
fi
#################启动训练脚本#################
export RANK=$i
export LOCAL_RANK=$i
PID_START=$((KERNEL_NUM * LOCAL_RANK))
PID_END=$((PID_START + KERNEL_NUM - 1))
taskset -c $PID_START-$PID_END python3.7 examples/sequence_labeling/task_sequence_labeling_ner_crf.py \
        --local_rank $i \
        --train_epochs ${train_epochs} \
        --data_path ${data_path} \
        --workers ${workers} \
        --lr ${lr} \
        --warm_factor ${warm_factor} \
        --opt_level ${opt_level} > $cur_path/test/output/${i}/train_${i}.log 2>&1 &
done
wait
ASCEND_DEVICE_ID=0
##################获取训练数据################
# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 训练用例信息，不需要修改
BatchSize=${BATCH_SIZE}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK}'p'_'acc'

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
grep "FPS" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk '{print $6}' &> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_fps.log
FPS=`cat ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_fps.log | sort -n | tail -$((train_epochs-1)) | awk '{a+=$1} END {if (NR != 0) printf("%.3f",a/NR)}'`
total_training_time=`grep "FPS" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk '{print $4}' | awk '{a+=$1} END {if (NR != 0) printf("%.3f",a)}'`
total_eval_time=`grep "eval_time_cost" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "eval_time_cost:" '{print $2}' | awk '{a+=$1} END {if (NR != 0) printf("%.3f",a)}'`
steps=`grep "FPS" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "/" 'END {print $1}' `
min_step_time=`grep "step time" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "step time:" '{print $2}' | awk 'BEGIN {min = 65536} {if ($1+0 < min+0) min=$1} END {print min}'`
maximum=`awk -v bs=${BATCH_SIZE} -v ws=${WORLD_SIZE} -v mt=${min_step_time} 'BEGIN{print(bs*ws/mt)}'`
total_sample=`awk -v bs=${BATCH_SIZE} -v ws=${WORLD_SIZE} -v te=${train_epochs} -v st=$steps 'BEGIN{print(bs*ws*te*st)}'`
train_average=`awk -v ts=${total_sample} -v ttt=$total_training_time 'BEGIN{print(ts/ttt)}'`

# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

# 输出训练精度,需要模型审视修改
train_accuracy=`grep -a "best_f1"  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'|awk -F "best_f1" '{print $NF}'|awk -F " " '{print $2}'`
# 打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

e2e_average=`awk -v ts=${total_sample} -v et=$e2e_time 'BEGIN{print(ts/et)}'`

# 性能看护结果汇总
# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${BATCH_SIZE}'*1000/'${FPS}'}'`

# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep Epoch: ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep -v Test|awk -F "Loss" '{print $NF}' | awk -F " " '{print $1}' >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

# 最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}'  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log

# perf report
echo "train_training_time : $total_training_time" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_perf_report.log
echo "train_eval_time : $total_eval_time" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_perf_report.log
echo "total_time : $e2e_time" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_perf_report.log
echo "training maximum images/sec : $maximum" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_perf_report.log
echo "training average images/sec : $train_average" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_perf_report.log
echo "end to end average images/sec : $e2e_average" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_perf_report.log
