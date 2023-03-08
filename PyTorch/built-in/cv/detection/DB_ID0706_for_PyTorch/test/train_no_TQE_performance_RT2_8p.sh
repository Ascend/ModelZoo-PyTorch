#!/bin/bash
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
################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE WORLD_SIZE MASTER_ADDR MASTER_PORT
# 网络名称，同目录名称
Network="DB_ID0706_for_PyTorch"

# 训练batch_size
batch_size=128

# 训练使用的npu卡数
export RANK_SIZE=8
export ENABLE_RUNTIME_V2=1

export WORLD_SIZE=8
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='18888'
export TASK_QUEUE_ENABLE=1
export DYNAMIC_OP="ADD"
# 数据集路径,保持为空,不需要修改
data_path=""
# 检验预训练模型的路径
model_path=$cur_path/path-to-model-directory

# 训练epoch
train_epochs=1

# 指定训练所使用的npu device卡id
device_id=6
bin=True
profiling=''
start_step=-1
stop_step=-1
# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --model_path* ]];then
        model_path=`echo ${para#*=}`
    elif [[ $para == --rt1 ]];then
        rt1=True
    elif [[ $para == --bin ]];then
        bin=True
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
    elif [[ $para == --start_step* ]];then
        start_step=`echo ${para#*=}`
    elif [[ $para == --stop_step* ]];then
        stop_step=`echo ${para#*=}`
    fi
done
# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
# 校验是否传入model_path不需要修改
if [[ $model_path == "" ]];then
    echo "[Error] para \"model_path\" must be confing"
    exit 1
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
# 使能rt1
if [ $rt1 ];then
    export ENABLE_RUNTIME_V2=0
    echo "use rt1 runtime"
fi
if [[ $profiling == "GE" ]];then
    export GE_PROFILING_TO_STD_OUT=1
fi
#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi


#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)
# # 非平台场景时source 环境变量
# check_etp_flag=`env | grep etp_running_flag`
# etp_flag=`echo ${check_etp_flag#*=}`
# if [ x"${etp_flag}" != x"true" ];then
#     source ${test_path_dir}/env_npu.sh
# fi

unset PTCOPY_ENABLE
unset SCALAR_TO_HOST_MEM
unset COMBINED_ENABLE
unset HCCL_CONNECT_TIMEOUT
unset MOTD_SHOWN
unset DYNAMIC_OP
unset TASK_QUEUE_ENABLE
unset HCCL_WHITELIST_DISABLE


sed -i "s|./datasets|$data_path|g" experiments/seg_detector/base_ic15.yaml

kernel_num=$(nproc)

if [ ${kernel_num} -lt 95 ];then
    cpu_number=${kernel_num}
else
    cpu_number=95
fi

taskset -c 0-${cpu_number} nohup python3.7 -W ignore train.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml \
        --data_path ${data_path}/icdar2015 \
        --resume ${data_path}/db_ckpt/MLT-Pretrain-ResNet50 \
        --seed=515 \
        --distributed \
        --amp \
        --device_list "0,1,2,3,4,5,6,7" \
        --num_gpus 8 \
        --local_rank 0 \
        --dist_backend 'hccl' \
        --world_size 1 \
        --epochs ${train_epochs} \
        --batch_size ${batch_size} \
        --lr 0.056 \
        --addr $(hostname -I |awk '{print $1}') \
        --amp \
        --epochs ${train_epochs} \
        --Port 2950 \
        --bin ${bin} \
        --profiling "${profiling}" \
        --start_step ${start_step} \
        --stop_step ${stop_step}  > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
FPS=`grep -a 'FPS@all' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print}' | awk -F '[#@all]' '{print $NF}'`
FPS=${FPS#* }  # 去除前面的空格字符
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

# 打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_RT2_bs${BatchSize}_${RANK_SIZE}'p'_noeditor_'perf'

# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep -a 'Epoch:' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F 'Loss' '{print $NF}' | awk '{print $1}' > ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

# 最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log