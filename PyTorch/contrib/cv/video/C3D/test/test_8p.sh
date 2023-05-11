#!/usr/bin/env bash
################�������ò�������Ҫģ�������޸�##################
# ��ѡ�ֶ�(�����ڴ˴�����Ĳ���): Network batch_size RANK_SIZE
# �������ƣ�ͬĿ¼����
Network="C3D"
# ָ��ѵ����ʹ�õ�npu device��id
device_id=0

for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    fi
done

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

#################����ѵ���ű�#################
# ѵ����ʼʱ�䣬����Ҫ�޸�
start_time=$(date +%s)
# ��ƽ̨����ʱsource ��������
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

#################������־���Ŀ¼������Ҫ�޸�#################
mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID

eval_file=$(find ./ -name "best_top1_acc_epoch*")
echo "==== Eval top accuracy of epoch pth is ${eval_file}"

python3 tools/test.py configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py ${eval_file} --eval top_k_accuracy > ${test_path_dir}/output/$ASCEND_DEVICE_ID/test_$ASCEND_DEVICE_ID.log 2>&1
wait
##################��ȡѵ������################
# ѵ������ʱ�䣬����Ҫ�޸�
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

train_accuracy=`grep 'top1_acc' ${test_path_dir}/output/$ASCEND_DEVICE_ID/test_$ASCEND_DEVICE_ID.log| tail -1 | awk -F 'top1_acc: ' '{print $2}'`


# �����ӡ������Ҫ�޸�
echo "------------------ Final result ------------------"
# ��ӡ������Ҫ�޸�
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"