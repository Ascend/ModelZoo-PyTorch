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

# 指定训练所使用的npu device卡id
device_id=0

#集合通信参数,不需要修改
export WORLD_SIZE=8


# 数据集路径,保持为空,不需要修改
data_path=""

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="MaskRCNN_for_PyTorch"
#训练batch_size
batch_size=16

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

# 配置数据集路径
if [ -d ${cur_path}/datasets ];
  then
    echo "${cur_path}/datasets exists."
  else
    mkdir -p ${cur_path}/datasets
fi
ln -s $data_path ${cur_path}/datasets/coco

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

#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source  ${test_path_dir}/env_npu.sh
fi

cd $cur_path

#训练开始时间，不需要修改
start_time=$(date +%s)
KERNEL_NUM=$(($(nproc)/8))
#################启动训练脚本#################
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

if [ -e ${cur_path}/last_checkpoint ];
  then
    echo "last_checkpoint exists, please delete or remove it."
    exit 1
fi

for i in $(seq 0 7)
  do
  if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];
    then
        echo "${cur_path}/test/output/${ASCEND_DEVICE_ID} exist."
    else
        mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
  fi
  export RANK=$i
  export LOCAL_RANK=$i
  
  let p_start=0+24*i
  let p_end=23+24*i

  taskset -c $p_start-$p_end python3 -u tools/train_net.py \
    --local_rank $i \
    --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
    DATALOADER.NUM_WORKERS 12 \
    SOLVER.MAX_ITER 95000 \
    SOLVER.CHECKPOINT_PERIOD 5000 \
    SAVE_CHECKPOINTS True \
    > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${i}.log 2>&1 &
  done

wait
#删除数据集软链接
rm -rf ${cur_path}/datasets/coco

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS_0=`grep "maskrcnn_benchmark.trainer INFO: eta:" $test_path_dir/output/${ASCEND_DEVICE_ID}/train_0.log | awk -F "average_train_fps:" '{print $2}' | awk 'END {print $NF}'`
FPS=`grep FPS $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'NR==2'|awk '{print $3}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep "Average Precision" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_0.log|awk -F "=" '{print $NF}'|head -1`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'acc'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep total_loss $test_path_dir/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F 'total_loss: ' '{print $2}'|awk '{print $1}' > $test_path_dir/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $test_path_dir/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
