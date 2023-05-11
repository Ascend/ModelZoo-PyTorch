#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="siamfc"
# 训练batch_size
batch_size=32
# 训练使用的npu卡数
export RANK_SIZE=8
# 数据集路径,修改为本地数据集路径
data_path="./data/OTB"
pth_path="./models/siamfc_50.pth"

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi


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


#################创建日志输出目录，不需要修改#################
ASCEND_DEVICE_ID=0
#if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
#    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
#    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
#else
#    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
#fi


#################启动训练脚本#################
# 测试开始时间，不需要修改
start_time=$(date +%s)
# source 环境变量
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi
# source ${test_path_dir}/set_env.sh

nohup python3 ./bin/my_test.py \
	--model_path ${pth_path} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/test_${ASCEND_DEVICE_ID}.log

wait
# 测试结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出训练精度,需要模型审视修改
eval_accuracy_prec=`grep prec_score: ${test_path_dir}/output/${ASCEND_DEVICE_ID}/test_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $4}'`
eval_accuracy_succ=`grep succ_score: ${test_path_dir}/output/${ASCEND_DEVICE_ID}/test_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $6}'`
eval_accuracy_succ_rate=`grep succ_rate: ${test_path_dir}/output/${ASCEND_DEVICE_ID}/test_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $8}'`
# 打印，不需要修改
echo "Final Precision score: ${eval_accuracy_prec}"
echo "Final Success score: ${eval_accuracy_succ}"
echo "Final Success rate: ${eval_accuracy_succ_rate}"
echo "E2E Evaluation Duration sec : $e2e_time"

# 性能看护结果汇总
# 测试用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "EvaluateAccuracy-Precision score = ${eval_accuracy_prec}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "EvaluateAccuracy-Success score = ${eval_accuracy_succ}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log