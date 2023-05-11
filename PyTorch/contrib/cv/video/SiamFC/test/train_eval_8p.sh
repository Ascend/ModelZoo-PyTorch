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
data_path="./data/OTB"
pth_path="./models/siamfc_50.pth"

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
#if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
#    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
#    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
#else
#    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
#fi


#################����ѵ���ű�#################
# ���Կ�ʼʱ�䣬����Ҫ�޸�
start_time=$(date +%s)
# source ��������
# ��ƽ̨����ʱsource ��������
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi
# source ${test_path_dir}/set_env.sh

nohup python3 ./bin/my_test.py \
	--model_path ${pth_path} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/test_${ASCEND_DEVICE_ID}.log

wait
# ���Խ���ʱ�䣬����Ҫ�޸�
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# �����ӡ������Ҫ�޸�
echo "------------------ Final result ------------------"
# ���ѵ������,��Ҫģ�������޸�
eval_accuracy_prec=`grep prec_score: ${test_path_dir}/output/${ASCEND_DEVICE_ID}/test_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $4}'`
eval_accuracy_succ=`grep succ_score: ${test_path_dir}/output/${ASCEND_DEVICE_ID}/test_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $6}'`
eval_accuracy_succ_rate=`grep succ_rate: ${test_path_dir}/output/${ASCEND_DEVICE_ID}/test_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $8}'`
# ��ӡ������Ҫ�޸�
echo "Final Precision score: ${eval_accuracy_prec}"
echo "Final Success score: ${eval_accuracy_succ}"
echo "Final Success rate: ${eval_accuracy_succ_rate}"
echo "E2E Evaluation Duration sec : $e2e_time"

# ���ܿ����������
# ����������Ϣ������Ҫ�޸�
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

# �ؼ���Ϣ��ӡ��${CaseName}.log�У�����Ҫ�޸�
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "EvaluateAccuracy-Precision score = ${eval_accuracy_prec}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "EvaluateAccuracy-Success score = ${eval_accuracy_succ}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log