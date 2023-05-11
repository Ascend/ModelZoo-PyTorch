#!/bin/bash

source test/env_npu.sh;

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="LResNet100E"
# backbone mode
net_mode="ir_se"
# 网络深度
net_depth=100
# 权重路径
pth_path=""
# 数据集路径,保持为空,不需要修改
data_path=""

# 验证batch_size
batch_size=4096
# 加载数据进程数
workers=8
# 
finetune=0

# 指定是gpu 还是npu 还是cpu
device_type='npu'
# 验证使用的npu卡数
export RANK_SIZE=8

# 是否进行分布式验证
distributed=1
# 每个节点的gpu个数
gpus=1
# 主节点号
dist_rank=0
# 后台方式
backend="hccl"
# tcp
dist_url="127.0.0.1:15566"


# 参数校验，data_path,weights为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --pth_path* ]];then
        pth_path=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

# 校验是否传入weights,不需要修改
if [[ $pth_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi


###############指定验证脚本执行路径###############
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
if [ -d ${test_path_dir}/output/eval_${RANK_SIZE} ];then
    rm -rf ${test_path_dir}/output/eval_${RANK_SIZE}
    mkdir -p ${test_path_dir}/output/eval_${RANK_SIZE}
else
    mkdir -p ${test_path_dir}/output/eval_${RANK_SIZE}
fi

LOG_ROOT=${test_path_dir}/output/eval_${RANK_SIZE}
# 变量
export DETECTRON2_DATASETS=${data_path}
export PYTHONPATH=./:$PYTHONPATH

#################启动验证脚本#################
# 验证开始时间，不需要修改
start_time=$(date +%s)

KERNEL_NUM=$(($(nproc)/${RANK_SIZE}))
for((i=0;i<$((RANK_SIZE));i++));
  do
    PID_START=$((KERNEL_NUM*i))
    PID_END=$((PID_START+KERNEL_NUM-1))
    taskset -c ${PID_START}-${PID_END} \
        python3 eval.py \
            --net_mode ${net_mode} \
            --net_depth ${net_depth} \
            --weights ${pth_path} \
            --data_path ${data_path} \
            --batch_size ${batch_size} \
            --num_workers ${workers} \
            --finetune ${finetune} \
            --device_type ${device_type} \
            --device_id ${i} \
            --distributed ${distributed} \
            --backend ${backend} \
            --dist_url ${dist_url} \
            --gpus ${gpus} \
            --dist_rank ${dist_rank} >> ${LOG_ROOT}/eval.log 2>&1 &
  done

wait



##################获取验证数据################
# 性能看护结果汇总
# 验证用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

# 验证结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 结果打印，不需要修改
echo "------------------ Final result ------------------"

# 输出验证精度
eval_accuracy=`cat ${LOG_ROOT}/eval.log | grep 'lfw_accuracy:' | tail -n 1 | awk '{print $2}'`

# 打印，不需要修改
echo "Final Eval Accuracy : ${eval_accuracy}"
# 输出时间
echo "E2E Eval Duration sec : $e2e_time"

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${LOG_ROOT}/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${LOG_ROOT}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${LOG_ROOT}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${LOG_ROOT}/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${LOG_ROOT}/${CaseName}.log
echo "EvalAccuracy = ${eval_accuracy}" >> ${LOG_ROOT}/${CaseName}.log
echo "E2EEvalTime = ${e2e_time}" >>  ${LOG_ROOT}/${CaseName}.log
