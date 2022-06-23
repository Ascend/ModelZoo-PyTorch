#!/bin/bash

################�������ò�������Ҫģ�������޸�##################
# ��ѡ�ֶ�(�����ڴ˴�����Ĳ���): Network batch_size RANK_SIZE
# �������ƣ�ͬĿ¼����
Network="SRGAN"
# ѵ��batch_size
batch_size=64
# ѵ��ʹ�õ�npu����
export RANK_SIZE=1
# ���ݼ�·��,����Ϊ��,����Ҫ�޸�
data_path=""

# ѵ��epoch
train_epochs=100
# ָ��ѵ����ʹ�õ�npu device��id
device_id=0
# �������ݽ�����
workers=8

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
if [ ! -d ${test_path_dir}/output ];then
    mkdir -p ${test_path_dir}/output
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
echo "=============start training==================="
python3.7 ./train1p.py \
    --nproc=${workers} \
    --use_npu=True \
    --train_data_path=${data_path}/VOC2012/train \
    --val_data_path=${data_path}/VOC2012/val \
    --output_dir=${test_path_dir}/output \
    --num_epochs=${train_epochs} \
    --amp_level='O1' \
    --amp=True \
    --loss_scale_g=128 \
	--loss_scale_d=128 \
    --performance=True \
    --batch_size=${batch_size} > ${test_path_dir}/output/train_performance_1p.log

wait


##################��ȡѵ������################
#ѵ������ʱ�䣬����Ҫ�޸�
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))
# �������FPS����Ҫģ�������޸�
fps=`grep -a 'Fps:'  ${test_path_dir}/output/train_performance_1p.log|awk -F " " '{print $NF}'|awk 'END {print}'`
FPS=${fps%%[*}
#�����ӡ������Ҫ�޸�
echo "E2E Training Duration sec : $e2e_time"
echo "Final Performance images/sec : $FPS"