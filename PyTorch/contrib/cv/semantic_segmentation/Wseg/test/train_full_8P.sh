#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="Wseg_for_PyTorch"
# 训练batch_size
batch_size=16
# 训练使用的npu卡数
export RANK_SIZE=8
# 数据集路径,保持为空,不需要修改
data_path=""
# 训练epoch
train_epochs=24
## 指定训练所使用的npu device卡id
device_id=0
## 加载数据进程数
workers=8
## 学习率设置  8P为0.008，1P为0.001
LR=0.008

# 设置数据集参数
DS=pascal_voc
# 设置训练标记参数
EXP=train_8P_NPU
RUN_ID=v01

CONFIG=configs/voc_resnet38.yaml
FILELIST=data/val_voc.txt

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
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

yaml_path=${cur_path}/configs/voc_resnet38.yaml

data_path=\"${data_path}\"
sed -i 's#DATA_ROOT:.*$#DATA_ROOT: '$data_path'#' ${yaml_path}
sed -i 's#NUM_WORKERS:.*$#NUM_WORKERS: '$workers'#' ${yaml_path}
sed -i 's#NUM_EPOCHS:.*$#NUM_EPOCHS: '$train_epochs'#' ${yaml_path}
#sed -i 's#TRAIN:BATCH_SIZE:.*$#BATCH_SIZE: '$batch_size'#' ${yaml_path}
sed -i 's#LR:.*$#LR: '$LR'#' ${yaml_path}

echo "exchange confige!"
#wait
#################创建日志输出目录，不需要修改#################

if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

#删除上一次训练保存下来的模型文件
rm -rf ${cur_path}/snapshots/${DS}/${EXP}
rm -rf ${cur_path}/logs/${DS}/${EXP}
#################启动训练（train）脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

#
export MASTER_ADDR=localhost
export MASTER_PORT=1222
export TASK_QUEUE_ENABLE=0
export ASCEND_GLOBAL_LOG_LEVEL=3

for i in $(seq 0 7)
do
    if [ $(uname -m) = "aarch64" ]
    then
    let p_start=0+24*i
    let p_end=23+24*i
    TRAIN_LOG_FILE=${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_8P_${ASCEND_DEVICE_ID}_$i.log
    taskset -c $p_start-$p_end python3 train.py --dataset $DS --cfg configs/voc_resnet38.yaml --exp $EXP --run $RUN_ID --local_rank $i > $TRAIN_LOG_FILE 2>&1 &
    echo "LOG: $TRAIN_LOG_FILE"
    else
        python3 train.py --dataset $DS --cfg configs/voc_resnet38.yaml --exp $EXP --run $RUN_ID --local_rank $i > $TRAIN_LOG_FILE 2>&1 &
    fi
done

wait

TRAIN_LOG_FILE_0=${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_8P_${ASCEND_DEVICE_ID}_0.log

#################启动验证(infer)脚本#################

# 预测阶段参数
# 预测结果输出根目录
#OUTPUT_DIR=output
# 加载最佳模型文件
model_id=`grep -a 'Saving checkpoint with score'  $TRAIN_LOG_FILE_0`
best_epoch=${model_id:0-2:2}
best_score=${model_id:0-18:4}
md_id=${model_id:0:5}
echo $model_id
echo $best_epoch
echo $best_score
SNAPSHOT=e0${best_epoch}Xs0.${best_score:0-4:1}${best_score:0-2:2}
# 无需填写
EXTRA_ARGS=

# limiting threads
NUM_THREADS=6

set OMP_NUM_THREADS=$NUM_THREADS
export OMP_NUM_THREADS=$NUM_THREADS

## Code goes here

#LISTNAME=`basename $FILELIST .txt`
SAVE_DIR=${test_path_dir}/output/${ASCEND_DEVICE_ID}/infer_8P_${ASCEND_DEVICE_ID}
LOG_FILE=${test_path_dir}/output/${ASCEND_DEVICE_ID}/infer_8P_${ASCEND_DEVICE_ID}.log
##echo 1
#
python3 infer_val.py --dataset $DS \
                         --cfg $CONFIG \
                         --exp $EXP \
                         --run $RUN_ID \
                         --resume $SNAPSHOT \
                         --infer-list $FILELIST \
                         --workers $NUM_THREADS \
                         --mask-output-dir $SAVE_DIR \
                         --local_rank $ASCEND_DEVICE_ID \
                         $EXTRA_ARGS > $LOG_FILE 2>&1 &
wait
##################计算并获取精度################

LISTNAME=`basename $FILELIST .txt`

# without CRF
data_path_len=${#data_path}
data_path=${data_path:1:data_path_len-2}
python3 eval_seg.py --data $data_path --filelist $FILELIST --masks $SAVE_DIR > $SAVE_DIR.eval 2>&1 &
wait
echo "Log: ${SAVE_DIR}.eval"

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))
#echo $e2e_time
#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改

fps=`grep -a 'Im/Sec: '  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_8P_${ASCEND_DEVICE_ID}_0.log|awk '{print $17}'|awk 'END {print}'`
FPS=`awk 'BEGIN{printf "%.2f\n", 8.0*'${fps}'}'`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a 'mIoU: '  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/infer_8P_${ASCEND_DEVICE_ID}.eval|awk '{print $2}'|awk 'END {print}'`

#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改

BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

echo $CaseName
##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

echo ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_8P_${ASCEND_DEVICE_ID}_0.log

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep '| loss:' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_8P_${ASCEND_DEVICE_ID}_0.log|awk '{print $13}' >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}'  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log