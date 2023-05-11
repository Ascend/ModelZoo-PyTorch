#!/usr/bin/env bash
################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="C3D"
# 指定训练所使用的npu device卡id
device_id=0

for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    fi
done

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

#################启动训练脚本#################
# 训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

#################创建日志输出目录，不需要修改#################
mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID

eval_file=$(find ./ -name "best_top1_acc_epoch*")
echo "==== Eval top accuracy of epoch pth is ${eval_file}"

python3 tools/test.py configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py ${eval_file} --eval top_k_accuracy > ${test_path_dir}/output/$ASCEND_DEVICE_ID/test_$ASCEND_DEVICE_ID.log 2>&1
wait
##################获取训练数据################
# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

train_accuracy=`grep 'top1_acc' ${test_path_dir}/output/$ASCEND_DEVICE_ID/test_$ASCEND_DEVICE_ID.log| tail -1 | awk -F 'top1_acc: ' '{print $2}'`


# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"