#!/bin/bash

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

#集合通信参数,不需要修改
export RANK_SIZE=8
RANK_ID_START=0
export WORLD_SIZE=8

# 数据集路径,保持为空,不需要修改
data_path=""

#网络名称,同目录名称,需要模型审视修改
Network="YoloV3_ID1790_for_PyTorch"

#训练batch_size,,需要模型审视修改
batch_size=64

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#关闭INF/NAN
export INF_NAN_MODE_ENABLE=0
#参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --fp32* ]];then
        fp32=`echo ${para#*=}`
    elif [[ $para == --hf32* ]];then
        hf32=`echo ${para#*=}`
    elif [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    fi
    if [[ $para == --conda_name* ]];then
      conda_name=`echo ${para#*=}`
      echo "PATH TRAIN BEFORE: $PATH"
      source set_conda.sh --conda_name=$conda_name
      source activate $conda_name
      echo "PATH TRAIN AFTER: $PATH"
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source  ${test_path_dir}/env_npu.sh
fi

#进入训练脚本目录，需要模型审视修改
cd $cur_path

#设置环境变量，不需要修改
RANK_ID=0
echo "Device ID: $RANK_ID"
export RANK_ID=$RANK_ID
export ASCEND_DEVICE_ID=$RANK_ID
ASCEND_DEVICE_ID=$RANK_ID

#创建DeviceID输出目录，不需要修改
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/ckpt
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/ckpt
fi

# 绑核，不需要的绑核的模型删除，需要的模型审视修改
#let a=RANK_ID*12
#let b=RANK_ID+1
#let c=b*12-1
chmod +x ${cur_path}/tools/dist_train.sh

#训练开始时间，不需要修改
start_time=$(date +%s)

sed -i "s|data/coco/|$data_path/|g" configs/yolo/yolov3_d53_mstrain-608_273e_coco.py

#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
export RANK_SIZE=8

KERNEL_NUM=$(($(nproc)/8))
for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
do
    export RANK=$RANK_ID

    if [ $(uname -m) = "aarch64" ]
    then
        PID_START=$((KERNEL_NUM * RANK_ID))
        PID_END=$((PID_START + KERNEL_NUM - 1))
        taskset -c $PID_START-$PID_END python3 ./tools/train.py configs/yolo/yolov3_d53_320_273e_coco.py \
            --launcher pytorch \
            --cfg-options data.samples_per_gpu=${batch_size} \
            optimizer.lr=0.0032 \
            --precision_mode ${precision_mode} \
            --seed 0 \
            --local_rank 0 > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    else
        python3 ./tools/train.py configs/yolo/yolov3_d53_320_273e_coco.py \
            --launcher pytorch \
            --cfg-options data.samples_per_gpu=${batch_size} \
            optimizer.lr=0.0032 \
            --precision_mode ${precision_mode} \
            --seed 0 \
            --local_rank 0 $fp32 $hf32 > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    fi
done

wait

sed -i "s|$data_path/|data/coco/|g" configs/yolo/yolov3_d53_mstrain-608_273e_coco.py
#8p情况下仅0卡(主节点)有完整日志,因此后续日志提取仅涉及0卡
ASCEND_DEVICE_ID=0

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
time=`grep -a 'Epoch '  $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "time: " '{print $2}'|awk -F "," '{print $1}'|awk 'END {print}'|sed 's/.$//'`
FPS=`awk 'BEGIN{printf "%.2f\n", '${RANK_SIZE}'*'${batch_size}'/'${time}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a 'Precision' $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $NF}'|head -1`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
if [[ ${fp32} == "--fp32" ]];then
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'fp32'_'acc'
elif [[ ${hf32} == "--hf32" ]];then
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'hf32'_'acc'
else
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'
fi

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${time}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep -a 'Epoch' $test_path_dir/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep eta:|awk -F "loss: " '{print $NF}' | awk -F " " '{print $1}' >> $test_path_dir/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $test_path_dir/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt|sed 's/.$//'`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
#退出anaconda环境
if [ -n "$conda_name" ];then
    echo "conda $conda_name deactivate"
    conda deactivate
fi
