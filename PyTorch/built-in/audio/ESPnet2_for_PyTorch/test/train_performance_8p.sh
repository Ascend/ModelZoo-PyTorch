#!/bin/bash
source ./test/env_npu.sh

#当前路径,不需要修改
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#集合通信参数,不需要修改
export RANK_SIZE=8


#网络名称,同目录名称,需要模型审视修改
Network="ESPnet2_for_PyTorch"

#训练batch_size,,需要模型审视修改
batch_size=256

#训练起始stage
stage=1

#训练epoch, 不需要修改
epochs=5

# 指定训练所使用的npu device卡id, 暂不支持修改
device_id=0

for para in $*
do
    if [[ $para == --stage* ]];then
        stage=`echo ${para#*=}`
    fi
done


# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
if [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi

#创建DeviceID输出目录，不需要修改
if [ -d $test_path_dir/output/${ASCEND_DEVICE_ID} ];then
    rm -rf $test_path_dir/output/$ASCEND_DEVICE_ID
    mkdir -p $test_path_dir/output/$ASCEND_DEVICE_ID
else
    mkdir -p $test_path_dir/output/$ASCEND_DEVICE_ID
fi

asr_log=$cur_path/egs2/aishell/asr1/exp/asr_train_asr_conformer_raw_zh_char_sp/train.log
result=$cur_path/egs2/aishell/asr1/exp/asr_train_asr_conformer_raw_zh_char_sp/RESULTS.md


#################启动训练脚本#################

# 必要参数替换配置文件
cd $cur_path/egs2/aishell/asr1

start_time=$(date +%s)

nohup bash run.sh \
  --stage ${stage} \
  --asr_args "--max_epoch ${epochs}" \
  --ngpu 8 &

wait

##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
ITERS=`grep "epoch results:" $asr_log |head -n 1 | awk -F "total_count=" '{print$2}' | awk -F "," '{print$1}'`
MINUTIES=`grep "epoch results:" $asr_log |tail -n 1 | awk -F " time=" '{print$2}' | awk -F " " '{print$1}'`
FPS=`awk 'BEGIN{printf "%.2f",('$ITERS' / ('$MINUTIES' * 60))}'`

#输出训练精度,需要模型审视修改
dev_accuracy=`grep "valid.acc.ave/dev" ${result} | tail -n 1 | awk -F "|" '{print$5}'`
test_accuracy=`grep "valid.acc.ave/test" ${result} | tail -n 1 | awk -F "|" '{print$5}'`
train_accuracy=${dev_accuracy}' '${test_accuracy}

#打印，不需要修改
echo "Final Performance iters/sec : $FPS"

#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

#获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", 1/'${FPS}'}'`

#从asr log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep "epoch results:" $asr_log |tail -n 1 | awk -F " loss=" '{print$2}' | awk -F "," '{print$1}' > ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log