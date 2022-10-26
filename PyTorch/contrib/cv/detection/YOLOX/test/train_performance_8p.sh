#!/bin/bash
################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="Yolox-x"
# 训练batch_size
batch_size=128
# 训练使用的npu卡数
export RANK_SIZE=8
# 数据集路径,保持为空,不需要修改
data_path=""

# 训练epoch
train_epochs=5


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
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi


#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi
if [ ! -d  "./datasets/VOCdevkit"  ]; then
    ln -s ${data_path} ./datasets/VOCdevkit
fi

KERNEL_NUM=$(($(nproc)/8))
for i in $(seq 0 7)
do
ASCEND_DEVICE_ID=$i

#创建DeviceID输出目录，不需要修改
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/$ASCEND_DEVICE_ID
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

if [ $(uname -m) = "aarch64" ]
then
    PID_START=$((KERNEL_NUM * i))
    PID_END=$((PID_START + KERNEL_NUM - 1))
    taskset -c $PID_START-$PID_END nohup python3.7 -u tools/train.py -n yolox-s \
        -f exps/example/yolox_voc/yolox_voc_s.py \
        -b ${batch_size} \
        -d 8 \
        --maxx_epoch ${train_epochs} \
        --use_npu \
        --device_id $i > ${test_path_dir}/output/${i}/train_${i}.log 2>&1 &
else
    nohup python3.7 -u tools/train.py -n yolox-s \
        -f exps/example/yolox_voc/yolox_voc_s.py \
        -b ${batch_size} \
        -d 8 \
        --maxx_epoch ${train_epochs} \
        --use_npu \
        --device_id $i > ${test_path_dir}/output/${i}/train_${i}.log 2>&1 &
fi
done


wait


##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修
FPS=`grep -a 'iter_time'  ${test_path_dir}/output/0/train_0.log|awk '/iter_time/{print '${batch_size}'/$15}'| sed 's/.$//' | sed 's/.$//'| awk '$0 >a{a=$0}END{print a}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`
#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep epoch: ${test_path_dir}/output/0/train_0.log|awk '{print $18,  $19, $20, $21, $22, $23, $24, $25, $26, $27}' >>  ${test_path_dir}/output/0/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}'  ${test_path_dir}/output/0/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/0/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/0/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/0/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/0/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/0/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/0/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${test_path_dir}/output/0/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${test_path_dir}/output/0/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/0/${CaseName}.log
