#!/bin/bash

#集合通信参数,不需要修改
export RANK_SIZE=1
export JOB_ID=10087
export ENABLE_RUNTIME_V2=1
# export GE_PROFILING_TO_STD_OUT=1
# export ASCEND_GLOBAL_LOG_LEVEL=1
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
export NPU_CALCULATE_DEVICE=$ASCEND_DEVICE_ID

#conda环境的名称
conda_name=py2

# 数据集路径,保持为空,不需要修改
data_path=""

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="C3D_ID4116_for_PyTorch"
#训练epoch
epochs=1
#训练batch_size
batch_size=64

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --epochs* ]];then
        epochs=`echo ${para#*=}`
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

#校验是否指定了device_id，分动态分配device_id与手动指定device_id，此处不需要修改
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi

###############指定训练脚本执行路径###############
#cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#创建DeviceID输出目录，不需要修改
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/$ASCEND_DEVICE_ID
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

cp ${data_path}/checkpoints/c3d* /root/.cache/torch/hub/checkpoints/

#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)

cur_path=`pwd`

#mmcv modify
location=`pip3 show mmcv | grep Location | awk '{print $NF}'`
cp ${cur_path}/additional_need/mmcv/distributed.py ${location}/mmcv/parallel/
cp ${cur_path}/additional_need/mmcv/test.py ${location}/mmcv/engine/
cp ${cur_path}/additional_need/mmcv/dist_utils.py ${location}/mmcv/runner/
cp ${cur_path}/additional_need/mmcv/optimizer.py ${location}/mmcv/runner/hooks/
cp ${cur_path}/additional_need/mmcv/epoch_based_runner.py ${location}/mmcv/runner/

#config modify
sed -i "s|data_root = 'data/ucf101/rawframes/'|data_root= '${data_path}/rawframes/'|g" ${cur_path}/configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb_1p.py
sed -i "s|data_root_val = 'data/ucf101/rawframes/'|data_root_val= '${data_path}/rawframes/'|g" ${cur_path}/configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb_1p.py
sed -i "s|ann_file_train = f'data/ucf101/ucf101_train_split_{split}_rawframes.txt'|ann_file_train= f'${data_path}/ucf101_train_split_{split}_rawframes.txt'|g" ${cur_path}/configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb_1p.py
sed -i "s|ann_file_val = f'data/ucf101/ucf101_val_split_{split}_rawframes.txt'|ann_file_val= f'${data_path}/ucf101_val_split_{split}_rawframes.txt'|g" ${cur_path}/configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb_1p.py
sed -i "s|ann_file_test = f'data/ucf101/ucf101_val_split_{split}_rawframes.txt'|ann_file_test= f'${data_path}/ucf101_val_split_{split}_rawframes.txt'|g" ${cur_path}/configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb_1p.py


python3 train.py ./configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb_1p.py \
    --validate \
    --seed 0 \
    --deterministic \
    --bin \
    --cfg-options data.workers_per_gpu=8 log_config.interval=20\
    --rank_id=$ASCEND_DEVICE_ID > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
avg_step_time=`grep "Epoch" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "time:" '{print $2}' | awk -F " " '{print $1}' | awk -F "," '{print $1}'| awk 'BEGIN{count=0}{if (NR>1 && NR<7){sum+=$1;count+=1}}END{printf "%.4f\n", sum/count}'`
FPS=`awk 'BEGIN{printf "%.4f\n", '$batch_size'/'$avg_step_time'}'`
#FPS=`grep -rn "wps=" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "wps=" '{print $2}' | awk -F "," '{print $1}' | awk '{if(NR>=325){print}}' | awk 'END {print}' |sed s/[[:space:]]//g`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
#TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
cat ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | grep "grad_norm" | awk '{print $(NF-2)}' | awk -F ',' '{print $1}' > ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

train_accuracy=`cat ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | grep top1_acc | awk END'{print $(NF-4)}' | awk -F ',' '{print $1}'`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${avg_step_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log

