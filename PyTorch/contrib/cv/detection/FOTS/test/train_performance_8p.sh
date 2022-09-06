#!/bin/bash

################�������ò�������Ҫģ�������޸�##################
# ��ѡ�ֶ�(�����ڴ˴�����Ĳ���): Network batch_size RANK_SIZE
# �������ƣ�ͬĿ¼����
Network="FOTS"
# ѵ��batch_size
batch_size=24
# ѵ��ʹ�õ�npu����
export RANK_SIZE=8
# ���ݼ�·��,����Ϊ��,����Ҫ�޸�
data_path=""

# ѵ��epoch
train_epochs=3
# ָ��ѵ����ʹ�õ�npu device��id
device_id=($(seq 0 7))
# �������ݽ�����
KERNEL_NUM=$(($(nproc)/8))

export MASTER_ADDR=localhost
export MASTER_PORT=22111
export HCCL_WHITELIST_DISABLE=1

KERNEL_NUM=$(($(nproc)/8))

NPUS=($(seq 0 7))
export NPU_WORLD_SIZE=${#NPUS[@]}
rank=0

# ����У�飬data_pathΪ�ش�������������������ɾ��ģ������������˴������������������ж��岢��ֵ
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# У���Ƿ���data_path,����Ҫ�޸�
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
# У���Ƿ�ָ����device_id,�ֶ�̬����device_id���ֶ�ָ��device_id,�˴�����Ҫ�޸�
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is 0-7"
else
    "[Error] device id must be config"
    exit 1
fi



###############ָ��ѵ���ű�ִ��·��###############
# cd����test�ļ���ͬ�㼶Ŀ¼��ִ�нű�����߼����ԣ�test_path_dirΪ����test�ļ��е�·��
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi


#################������־���Ŀ¼������Ҫ�޸�#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output
    mkdir -p ${test_path_dir}/output
else
    mkdir -p ${test_path_dir}/output
fi


#################����ѵ���ű�#################
#ѵ����ʼʱ�䣬����Ҫ�޸�
start_time=$(date +%s)
# ��ƽ̨����ʱsource ��������
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/set_npu_env.sh
fi

for i in ${NPUS[@]}
do
    export NPU_CALCULATE_DEVICE=${i}
    export RANK=${rank}
    echo run process ${rank}
    
    PID_START=$((KERNEL_NUM * RANK))
    PID_END=$((PID_START + KERNEL_NUM - 1))

    time taskset -c $PID_START-$PID_END python3 ./train_8p.py --train-folder ${data_path} --continue-training --pf --batch-size ${batch_size} --batches-before-train 2\
    --num-workers $KERNEL_NUM > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    let rank++
done

wait

#ѵ������ʱ�䣬����Ҫ�޸�
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))


#�����ӡ������Ҫ�޸�
echo "------------------ Final result ------------------"
#�������FPS����Ҫģ�������޸�
FPS=`grep FPS ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $NF}'|awk '{sum+=$1} END {print  sum/NR}'`

#��ӡ������Ҫ�޸�
echo "Final Performance images/sec : $FPS"

#��ӡ������Ҫ�޸�
echo "E2E Training Duration sec : $e2e_time"

#���ܿ����������
#ѵ��������Ϣ������Ҫ�޸�
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##��ȡ�������ݣ�����Ҫ�޸�
#������
ActualFPS=${FPS}
#������ѵ��ʱ��
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#��train_$ASCEND_DEVICE_ID.log��ȡLoss��train_${CaseName}_loss.txt�У���Ҫ����ģ������
grep 'Mean loss' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F 'loss=' '{print $NF}' | awk -F ']' '{print $1}' > ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#���һ������lossֵ������Ҫ�޸�
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#�ؼ���Ϣ��ӡ��${CaseName}.log�У�����Ҫ�޸�
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log