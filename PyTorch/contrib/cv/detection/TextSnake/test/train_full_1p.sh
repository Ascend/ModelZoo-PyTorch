#!/bin/bash

################�������ò�������Ҫģ�������޸�##################
# ��ѡ�ֶ�(�����ڴ˴�����Ĳ���): Network batch_size RANK_SIZE
# �������ƣ�ͬĿ¼����
Network="TextSnake"
# ѵ��batch_size
batch_size=32
# ѵ��ʹ�õ�npu����
export RANK_SIZE=1

data_path="total-text"

# ѵ��epoch
train_epochs=200
# ָ��ѵ����ʹ�õ�npu device��id
device_id=0
# �������ݽ�����
workers=128

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
    echo "device id is ${ASCEND_DEVICE_ID}"
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
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi


#################����ѵ���ű�#################
#ѵ����ʼʱ�䣬����Ҫ�޸�
start_time=$(date +%s)
# ��ƽ̨����ʱsource ��������
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi
export HCCL_CONNECT_TIMEOUT=1800
export WORLD_SIZE=$RANK_SIZE
python3.7 -u train_textsnake_npu.py \
    example \
    --dataset=${data_path} \
    --num_workers=${workers} \
    --learning-rate=0.1 \
    --device='npu' \
    --gpu=${ASCEND_DEVICE_ID} \
    --viz \
    --max_epoch 200 \
    --batch_size=${batch_size} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

##################��ȡѵ������################
#ѵ������ʱ�䣬����Ҫ�޸�
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#�����ӡ
echo "------------------ Final result ------------------"
#�������FPS
FPS=`grep -a 'FPS'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $11}'|awk 'END {print}'`
#��ӡ
echo "Final Performance images/sec : $FPS"

#���ѵ������
train_accuracy=`grep -a '* Acc@1'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'|awk -F "Acc@1" '{print $NF}'|awk -F " " '{print $1}'`
#��ӡ
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#���ܿ����������
#ѵ��������Ϣ������Ҫ�޸�
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##��ȡ�������ݣ�����Ҫ�޸�
#������
ActualFPS=${FPS}
#������ѵ��ʱ��
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#��train_$ASCEND_DEVICE_ID.log��ȡLoss��train_${CaseName}_loss.txt��
grep Epoch: ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep -v Test|awk -F "Loss" '{print $NF}'|awk -F " " '{print $1}' >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#���һ������lossֵ������Ҫ�޸�
ActualLoss=`awk 'END {print}'  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#�ؼ���Ϣ��ӡ��${CaseName}.log�У�����Ҫ�޸�
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



