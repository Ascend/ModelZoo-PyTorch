#!/bin/bash

################�������ò�������Ҫģ�������޸�##################
# ��ѡ�ֶ�(�����ڴ˴�����Ĳ���): Network batch_size RANK_SIZE
# �������ƣ�ͬĿ¼����
Network="siamfc"
# ѵ��batch_size
batch_size=32
# ѵ��ʹ�õ�npu����
export RANK_SIZE=8
# ���ݼ�·��,�޸�Ϊ�������ݼ�·��
data_path="./data/ILSVRC_VID_CURATION"

# ѵ��epoch
train_epochs=1

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
if [ -d ${test_path_dir}/output/8p_perf/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/8p_perf/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/8p_perf/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/8p_perf/$ASCEND_DEVICE_ID
fi


#################����ѵ���ű�#################
# ѵ����ʼʱ�䣬����Ҫ�޸�
start_time=$(date +%s)
# source ��������
# ��ƽ̨����ʱsource ��������
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi
# source ${test_path_dir}/set_env.sh

#nohup python3.7 -m torch.distributed.launch --nproc_per_node 8 --master_port 22331  ./bin/my_train.py \
#	--world_size 8 > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log


RANK_ID_START=0
RANK_SIZE=8

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do

KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))


nohup \
taskset -c $PID_START-$PID_END python3.7 -u ./bin/my_train.py \
	--data ${data_path} \
	--workers $(($(nproc)/8)) \
	--local_rank $RANK_ID \
	--world_size 8 \
	--epoch ${train_epochs} > ${test_path_dir}/output/8p_perf/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log &

done

wait
# ѵ������ʱ�䣬����Ҫ�޸�
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# �����ӡ������Ҫ�޸�
echo "------------------ Final result ------------------"
# �������FPS����Ҫģ�������޸�
FPS=`grep -a 'FPS'  ${test_path_dir}/output/8p_perf/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $NF}'|awk 'END {print}'`
# ��ӡ������Ҫ�޸�
echo "Final Performance images/sec : $FPS"
echo "E2E Training Duration sec : $e2e_time"

# ���ܿ����������
# ѵ��������Ϣ������Ҫ�޸�
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'fps_loss'

# ��ȡ�������ݣ�����Ҫ�޸�
# ������
ActualFPS=${FPS}
# ������ѵ��ʱ��
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

ActualLoss=`grep EPOCH ${test_path_dir}/output/8p_perf/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep -v Test|awk -F "train_loss" '{print $NF}' | awk -F " " '{print $2}'`

# �ؼ���Ϣ��ӡ��${CaseName}.log�У�����Ҫ�޸�
echo "Network = ${Network}" > ${test_path_dir}/output/8p_perf/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/8p_perf/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/8p_perf/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/8p_perf/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/8p_perf/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/8p_perf/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/8p_perf/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/8p_perf/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/8p_perf/$ASCEND_DEVICE_ID/${CaseName}.log