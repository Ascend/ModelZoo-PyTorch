#!/bin/bash

################�������ò�������Ҫģ�������޸�##################
# ��ѡ�ֶ�(�����ڴ˴�����Ĳ���): Network batch_size RANK_SIZE
# �������ƣ�ͬĿ¼����
Network="ResNet1202_ID4111_for_PyTorch"
# ѵ��batch_size
batch_size=128
# ѵ��ʹ�õ�npu����
export RANK_SIZE=8
# ģ�ͽṹ
arch="resnet1202"
# ���ݼ�·��,����Ϊ��,����Ҫ�޸�
data_path=""

# ѵ��epoch 200
train_epochs=1
# �������ݽ�����
workers=128

# ����У�飬data_pathΪ�ش�������������������ɾ��ģ������������˴������������������ж��岢��ֵ
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --arch* ]];then
        arch=`echo ${para#*=}`
    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
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


#################������־���Ŀ¼������Ҫ�޸�#################
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi


#################����ѵ���ű�#################
# ѵ����ʼʱ�䣬����Ҫ�޸�
start_time=$(date +%s)
# ��ƽ̨����ʱsource ��������
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
else
  #etpƽ̨���ݼ�����
  mkdir -p ${cur_path}/data
  ln -nsf $ckpt_path/cifar-10-python.tar.gz ${cur_path}/data
  ln -nsf $data_path/* ${cur_path}/data
  data_path=${cur_path}/data
  export ONLY_TRAIN=True
fi

export NODE_RANK=0

nohup python3.7 ./DistributedResnet/main_apex_npu.py \
        --arch ${arch} \
        --data ${data_path} \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49 \
        --workers=${workers} \
        --learning-rate=0.05 \
        --warmup=8 \
        --label-smoothing=0.1 \
        --mom=0.9 \
        --weight-decay=1.0e-04  \
        --static-loss-scale=128 \
        --combine-ddp \
        --print-freq=1 \
        --dist-url='tcp://127.0.0.1:50000' \
        --dist-backend='hccl' \
        --multiprocessing-distributed \
        --world-size=1 \
        --rank=0 \
        --benchmark=0 \
        --device='npu' \
        --epochs=${train_epochs} \
        --batch-size=${batch_size} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait

##################��ȡѵ������################
# ѵ������ʱ�䣬����Ҫ�޸�
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# ѵ��������Ϣ������Ҫ�޸�
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

# �����ӡ������Ҫ�޸�
echo "------------------ Final result ------------------"
# �������FPS����Ҫģ�������޸�
grep "FPS@all" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk '{print $11}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_fps.log
FPS=`cat ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_fps.log | awk '{a+=$1} END {if (NR != 0) printf("%.3f",a/NR)}'`
# ��ӡ������Ҫ�޸�
echo "Final Performance images/sec : $FPS"

# ���ѵ������,��Ҫģ�������޸�
#train_accuracy=`grep -a '* Acc@1'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'|awk -F "Acc@1" '{print $NF}'|awk -F " " '{print $1}'`
# ��ӡ������Ҫ�޸�
#echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

# ���ܿ����������
# ��ȡ�������ݣ�����Ҫ�޸�
# ������
ActualFPS=${FPS}
# ������ѵ��ʱ��
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

# ��train_$ASCEND_DEVICE_ID.log��ȡLoss��train_${CaseName}_loss.txt�У���Ҫ����ģ������
grep Epoch: ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep -v Test|awk -F "Loss" '{print $NF}' | awk -F " " '{print $1}' >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

# ���һ������lossֵ������Ҫ�޸�
ActualLoss=`awk 'END {print}'  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

# �ؼ���Ϣ��ӡ��${CaseName}.log�У�����Ҫ�޸�
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log