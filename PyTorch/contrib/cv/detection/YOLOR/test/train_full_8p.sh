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
#网络名称
Network="YOLOR"

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

ln -snf $data_path ./data/coco

# 指定训练所使用的npu device卡id
device_id=0

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
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/ckpt
    mkdir -p ${test_path_dir}/output/overflow_dump
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/ckpt
    mkdir -p ${test_path_dir}/output/overflow_dump
fi
#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source  ${test_path_dir}/env_npu.sh
fi

#训练开始时间，不需要修改
start_time=$(date +%s)
# addr is the ip of training server
if [ $(uname -m) = "aarch64" ]
then
	for i in $(seq 0 7)
	do 
	let p_start=0+24*i
	let p_end=23+24*i
	taskset -c $p_start-$p_end $CMD python3 train_mp.py \
        --cfg cfg/yolor_p6.cfg \
        --data data/coco.yaml \
        --addr 127.0.0.1 \
        --weights '' \
        --batch-size 64 \
        --img 1280 1280 \
        --local_rank $i \
        --device npu \
        --device-num 8 \
        --name yolor_p6_npu_8p_full \
        --hyp hyp.scratch.1280.yaml \
        --epochs 300 \
        --full > $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
	done
else
   python3 train.py \
        --cfg cfg/yolor_p6.cfg \
        --data data/coco.yaml \
        --addr 127.0.0.1 \
        --weights '' \
        --batch-size 64 \
        --img 1280 1280 \
        --local_rank 0 \
        --device npu \
        --device-num 8 \
        --name yolor_p6_npu_8p_full \
        --hyp hyp.scratch.1280.yaml \
        --epochs 300 \
        --full > $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
fi
wait

acc=`grep -a 'IoU=0.50:0.95' npu_8p_full.log | grep 'Average Precision'|awk 'NR==1'| awk -F " " '{print $13}'`
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#训练用例信息，不需要修改
RANK_SIZE=8
BatchSize=64
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

FPS=`grep "FPS:" $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "FPS:" '{print $2}'|awk -F ']' '{print $1}' |tail -1`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
echo "Final Accuracy : $acc"

#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

TrainingTime=`echo "scale=2;${BatchSize} / ${FPS}"|bc`

ActualLoss=`grep "totalLoss:" $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $7}' |tr -d "totalLoss:"`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${FPS}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
