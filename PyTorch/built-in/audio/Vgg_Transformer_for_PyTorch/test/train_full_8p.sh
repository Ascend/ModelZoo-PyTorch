#!/bin/bash

# 当前路径,不需要修改
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

# 集合通信参数,不需要修改
export RANK_SIZE=8
export WORLD_SIZE=$RANK_SIZE
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='29668'

# 网络名称,同目录名称,需要模型审视修改
Network="Vgg_Transformer_for_PyTorch"
# 数据集路径,保持为空,不需要修改
data_path=""
# 训练epoch
train_epochs=80
# 训练batch_size,,需要模型审视修改
token_size=5000

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --train_epochs* ]];then
        train_epochs=`echo ${para#*=}`
    elif [[ $para == --token_size* ]];then
        token_size=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

##################创建日志输出目录，根据模型审视##################
# 模型采用非循环方式启动多卡训练，创建日志输出目录如下；采用循环方式启动多卡训练的模型，在循环中创建日志输出目录，可参考CRNN模型
# 非循环方式下8卡训练日志输出路径中的ASCEND_DEVICE_ID默认为0，只是人为指定文件夹名称，不涉及训练业务
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/$ASCEND_DEVICE_ID ];then
    rm -rf ${test_path_dir}/output/*
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

##############执行训练##########
# 训练开始时间，不需要修改
start_time=$(date +%s)

# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

# 使用NPU融合优化器开关, True为开启, False为关闭.
export NPU_FUSED_ENABLE=True

KERNEL_NUM=$(($(nproc)/8))

for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
do
  export RANK=$RANK_ID
  if [ -d ${test_path_dir}/output/${RANK_ID} ];then
    rm -rf ${test_path_dir}/output/${RANK_ID}
    mkdir -p ${test_path_dir}/output/${RANK_ID}
  else
    mkdir -p ${test_path_dir}/output/${RANK_ID}
  fi

  PID_START=$((KERNEL_NUM * RANK_ID))
  PID_END=$((PID_START + KERNEL_NUM - 1))
  nohup taskset -c $PID_START-$PID_END python3 train.py ${data_path} \
                --distributed-world-size 8 --device-id $RANK_ID --distributed-rank $RANK_ID --distributed-no-spawn \
                --save-dir ${test_path_dir}/output/saved_results --keep-last-epochs 3 \
                --max-epoch ${train_epochs} --task speech_recognition --arch vggtransformer_2 \
                --optimizer adadelta --lr 1.0 \
                --adadelta-eps 1e-8 --adadelta-rho 0.95 --clip-norm 10.0 --max-tokens ${token_size} \
                --log-format json --log-interval 1 \
                --criterion cross_entropy_acc \
                --user-dir examples/speech_recognition/ --seed 1 \
                > ${test_path_dir}/output/${RANK_ID}/train_${RANK_ID}.log 2>&1 &
done
wait

export PYTHONPATH=$PYTHONPATH:/${cur_path}
nohup python3 examples/speech_recognition/infer.py ${data_path} \
        --task speech_recognition --max-tokens 25000 \
        --nbest 1 --path ${test_path_dir}/output/saved_results/checkpoint_last.pt \
        --beam 20 --results-path ${test_path_dir}/output/infer_results \
        --batch-size 40 --gen-subset test-clean --user-dir examples/speech_recognition/ \
        > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/eval_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
WPS=`grep -rn "| INFO | train_inner |" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "\"wps\": " '{print$2}' | awk -F "\"" '{print $2}' | tail -n+10 | awk '{sum+=$1} END {print"",sum/NR}' | sed s/[[:space:]]//g`
train_wall=`grep -rn "| INFO | train_inner |" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "\"train_wall\": " '{print$2}' | awk -F "\"" '{print $2}' | tail -n+10 | awk '{sum+=$1} END {print"",sum/NR}' | sed s/[[:space:]]//g`

# 打印，不需要修改
echo "Final Performance words/sec : $WPS"
echo "train_wall : $train_wall"

# 输出训练精度,需要模型审视修改
train_accuracy=`grep "INFO:__main__:WER:" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/eval_${ASCEND_DEVICE_ID}.log | awk -F "WER: " '{print $2}' | awk 'END {print}'`

# 打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : ${e2e_time}"

# 性能看护结果汇总
# 训练用例信息，不需要修改
TokenSize=${token_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${TokenSize}_${RANK_SIZE}'p'_'acc'

## 获取性能数据，不需要修改
# 吞吐量
ActualWPS=${WPS}
# 单迭代训练时长
TrainingTime=${train_wall}

# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep -rn "| INFO | train_inner |" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "\"loss\": " '{print$2}' | awk -F "\"" '{print $2}'  > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt

# 最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TokenSize = ${TokenSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualWPS = ${ActualWPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}">> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log