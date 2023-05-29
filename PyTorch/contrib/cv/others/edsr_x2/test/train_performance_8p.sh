#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="EDSR_x2"
# 训练batch_size
batch_size=16
# 训练使用的npu卡数
export RANK_SIZE=8
# 数据集路径,保持为空,不需要修改
data_path=""

# 训练epoch
train_epochs=2
# 学习率
learning_rate=1e-4
# 加载数据进程数
workers=184


# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --workers* ]];then
        workers=`echo ${para#*=}`
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
cur_path_last_diename=${cur_path##*/}
if [ x"${cur_path_last_diename}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

output_dir=${test_path_dir}/output/train_perf_8p
#################创建日志输出目录，不需要修改#################
ASCEND_DEVICE_ID=0
if [ -d ${output_dir} ];then
    rm -rf ${output_dir}
    mkdir -p ${output_dir}
else
    mkdir -p ${output_dir}
fi
output_log=${output_dir}/train_perf_8p.log

#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi
python3 main_8p.py \
	--dir_data ${data_path} \
	--save ${output_dir}/npu_8p \
	--seed 49 \
	--batch_size ${batch_size} \
	--amp \
	--device npu \
    --workers ${workers} \
	--loss_scale 128.0 \
	--opt_level 'O2' \
	--lr ${learning_rate} \
	--epochs ${train_epochs} \
	--world_size 1 \
	--dist-backend "hccl" > ${output_log} 2>&1 &

wait



##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a 'FPS'  ${output_log}|tail -1 |awk -F " " '{print $2}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a 'This Time' ${output_log} | tail -1 | awk -F " " '{print $5}'`
#打印，不需要修改
echo "Final Train Accuracy (PSNR): ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep Loss: ${output_log} | tail -1 | awk -F " " '{print $2}' >>  ${output_dir}/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}'  ${output_dir}/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${output_dir}/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${output_dir}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${output_dir}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${output_dir}/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${output_dir}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${output_dir}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${output_dir}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${output_dir}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${output_dir}/${CaseName}.log