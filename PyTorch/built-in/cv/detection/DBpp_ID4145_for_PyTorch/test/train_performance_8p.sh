#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="DB_ID4145_for_PyTorch"
# 训练使用的npu卡数
export BATCH_SIZE=24
export WORLD_SIZE=8
# 数据集路径,保持为空,不需要修改
data_path=""

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --conda_name* ]];then
        conda_name=`echo ${para#*=}`
        cur_path=`pwd`
        echo --$cur_path
        cd $cur_path
        source ${cur_path}/set_conda.sh --conda_name=$conda_name
        source activate $conda_name
        cd -
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
ASCEND_DEVICE_ID=0
#################创建日志输出目录，不需要修改#################
if [ -d ${cur_path}/data ];
then
        ln -snf $data_path ./data/
else
        mkdir -p ${cur_path}/data
        ln -snf $data_path ./data/
fi

if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];
then
        rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
else
        mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
fi
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi
# 加载ckpt
mkdir -p checkpoints/textdet/dbnetpp/
cp -r $data_path/res50dcnv2_synthtext.pth checkpoints/textdet/dbnetpp/
bash tools/dist_train.sh configs/textdet/dbnetpp/dbnetpp_r50dcnv2_fpnc_5e_icdar2015.py dbnet $WORLD_SIZE \
        > $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

##################获取训练数据################
# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))
# 训练用例信息，不需要修改
BatchSize=${BATCH_SIZE}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'_'perf'

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
grep -a "time:" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "time:" '{print substr($2,0,6)}' &> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_fps.log
FPS=`cat ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_fps.log | sort -n | head -8 | awk -v bs=$BATCH_SIZE -v ws=$WORLD_SIZE '{a+=$1} END {if (NR != 0) printf("%.3f", 1/a*NR*bs*ws)}'`
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${BATCH_SIZE}'*1000/'${FPS}'}'`

# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep -a Epoch ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "loss: " '{print $NF}' >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

# 最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}'  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log