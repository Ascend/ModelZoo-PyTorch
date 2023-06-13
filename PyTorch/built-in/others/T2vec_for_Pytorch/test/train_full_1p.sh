#!/bin/bash
Network="T2vec"
# 训练batch_size
batch_size=768
#训练steps
steps=0
# 训练使用的npu卡数
RANK_SIZE=1
# 数据集路径,保持为空,不需要修改
data_path=""
KERNEL_NUM=$(($(nproc)/8))
#参数校验，不需要修改                   ***********************************?
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --steps* ]];then
        steps=`echo ${para#*=}`
    fi
done
echo "data_path: $data_path"

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
##################指定训练脚本执行路径##################
# cd到与test文件同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi
echo "cur_path: ${cur_path}"
##################创建日志输出目录，根据模型审视##################              **********************************?
# 模型采用非循环方式启动多卡训练，创建日志输出目录如下；采用循环方式启动多卡训练的模型，在循环中创建日志输出目录，可参考CRNN模型
# 非循环方式下8卡训练日志输出路径中的ASCEND_DEVICE_ID默认为0，只是人为指定文件夹名称， 不涉及训练业务
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/$ASCEND_DEVICE_ID ];then
    rm -rf ${test_path_dir}/output/$ASCEND_DEVICE_ID
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi
echo "test_path_dir: ${test_path_dir}"
##################启动训练脚本##################
#训练开始时间，不需要修改
start_time=$(date +%s)
source ./test/env_npu.sh

nohup python3 -u t2vec.py \
             -vocab_size 18866 \
             -criterion_name "KLDIV" \
             -data ${data_path} \
             -checkpoint ${data_path}/checkpoint.pt \
             -knearestvocabs "${data_path}/porto-vocab-dist-cell100.h5" \
             -batch $batch_size \
             -max_step $steps \
             -local_rank $ASCEND_DEVICE_ID > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

abs_data_path=`realpath $data_path`
nohup julia evales.jl --data_path ${abs_data_path} \
        --pth_path ${abs_data_path}/best_model.pt >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait
##################获取训练数据##################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
# 提取整网性能
FPS=`grep -a 'Iteration:' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | grep FPS | awk -F "FPS@all:" '{print $2}' | awk -F ' ' '{print $1}' | tail -n 5 | awk '{sum+=$1;i+=1} END{print sum / i}'`
# 打印，不需要修改
echo "Final Performance image/sec : $FPS"
train_accuracy=`grep -a 'mean rank:' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | grep 'dbsize: 20000' | awk -F "mean rank: " '{print $2}' | awk -F ' ' '{print $1}'`
echo "Final Train Accuracy :" $train_accuracy
echo "E2E Training Duration sec : $e2e_time"


#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'
echo "CaseName: $CaseName"
# 获取性能数据
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${BatchSize}' * 1000 / '${ActualFPS}'}'`
echo "TrainingTime: $TrainingTime"

grep 'Generative Loss' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "Generative Loss:" '{print $2}' |awk -F " " '{print $1}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
# 最后一个迭代loss值
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`
# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log

