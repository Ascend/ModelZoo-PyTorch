#!/bin/bash

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
export RANK_SIZE=1

# 数据集路径,保持为空,不需要修改
data_path=""

#网络名称,同目录名称,需要模型审视修改
Network="ESPnet_for_PyTorch"

#训练batch_size,,需要模型审视修改
batch_size=32

#训练epoch, 不需要修改
epochs=2

# 指定训练所使用的npu device卡id
device_id=0

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

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

asr_log=$cur_path/egs/aishell/asr1/exp/train_sp_pytorch_train/results/log
test_result=$cur_path/egs/aishell/asr1/exp/train_sp_pytorch_train/decode_test_decode_lm_4/result.txt
dev_result=$cur_path/egs/aishell/asr1/exp/train_sp_pytorch_train/decode_dev_decode_lm_4/result.txt

rm -rf ${asr_log}
rm -rf ${test_result}
rm -rf ${dev_result}

#################启动训练脚本#################

# 必要参数替换配置文件
cd $cur_path/egs/aishell/asr1

start_time=$(date +%s)

nohup bash run.sh \
  --stage 4  \
  --stop_stage 4 \
  --ngpu 1 \
  --asr_test_epochs ${epochs} \
  --test_output_dir $test_path_dir/output \
  --data $data_path disown &

wait

##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

#输出性能FPS，需要模型审视修改
grep -rns '"elapsed_time":' ${asr_log} |awk '{print$3}' |awk -F "," '{print$1}' > ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_time.txt
time_7500=`sed -n '75p' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_time.txt`
time_14600=`sed -n '146p' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_time.txt`
FPS=`awk 'BEGIN{printf "%.4f\n", 7100/('${time_14600}'-'${time_7500}')}'`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", 1/'${FPS}'}'`

#从asr log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep -rns '"main/loss":' ${asr_log} |awk '{print$3}' |awk -F "," '{print$1}' > ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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