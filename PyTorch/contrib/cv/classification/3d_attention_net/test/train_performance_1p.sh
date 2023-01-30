#!/bin/bash

################�������ò�������Ҫģ�������޸�##################
# ��ѡ�ֶ�(�����ڴ˴�����Ĳ���): Network batch_size RANK_SIZE
# �������ƣ�ͬĿ¼����
Network="3d_attention_net_for_PyTorch"
# ѵ��batch_size
batch_size=512
# ѵ��ʹ�õ�npu����
RANK_SIZE=1
# ���ݼ�·��,����Ϊ��,����Ҫ�޸�
data_path=""
# ָ��ѵ����ʹ�õ�npu device��id
device_id=0
# ѵ��epoch
train_epochs=3
# ����У�飬data_pathΪ�ش�������������������ɾ��ģ������������˴������������������ж��岢��ֵ
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    fi
done

# У���Ƿ���data_path,����Ҫ�޸�
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
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

# У���Ƿ�ָ����device_id,�ֶ�̬����device_id���ֶ�ָ��device_id,�˴�����Ҫ�޸�
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi

#################������־���Ŀ¼������Ҫ�޸�#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi
#���ݼ�����
mkdir -p ${cur_path}/data
ln -nsf $data_path ${cur_path}/data/

#################����ѵ���ű�#################
# ѵ����ʼʱ�䣬����Ҫ�޸�
start_time=$(date +%s)
# ��ƽ̨����ʱsource ��������
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

#################����ѵ���ű�#################
# ѵ����ʼʱ�䣬����Ҫ�޸�
start_time=$(date +%s)
RANK_ID_START=0
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do

KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))

taskset -c $PID_START-$PID_END python3.7 train.py \
    --device_type="NPU" \
    --device_id=$ASCEND_DEVICE_ID \
    --device_num=$RANK_SIZE \
    --is_train="True" \
    --is_pretrain="False" \
    --num_classes=10 \
    --total_epochs=$train_epochs \
    --dist_url='tcp://127.0.0.1:49876' \
    --train_batch_size=$batch_size \
    --test_batch_size=128 > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done
wait
##################��ȡѵ������################
# ѵ������ʱ�䣬����Ҫ�޸�
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# ѵ��������Ϣ������Ҫ�޸�
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'full'

# �����ӡ������Ҫ�޸�
echo "------------------ Final result ------------------"
# �������FPS����Ҫģ�������޸�
FPS=`grep -a 'FPS'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'|awk -F "FPS: " '{print $NF}'`

# ��ӡ������Ҫ�޸�
echo "Final Performance images/sec : $FPS"
# ���ѵ������,��Ҫģ�������޸�
train_accuracy=`grep -a 'acc'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'|awk -F "acc, " '{print $NF}'`
# ��ӡ������Ҫ�޸�
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

# ������
ActualFPS=${FPS}

# �ؼ���Ϣ��ӡ��${CaseName}.log�У�����Ҫ�޸�
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log

