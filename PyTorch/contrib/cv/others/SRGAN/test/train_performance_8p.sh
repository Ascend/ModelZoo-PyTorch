#!/usr/bin/env bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="SRGAN"
# 训练batch_size
batch_size=64
# 训练使用的npu卡数
export RANK_SIZE=1
# 数据集路径,保持为空,不需要修改
data_path=""

# 训练epoch
train_epochs=3
# 指定训练所使用的npu device卡id
device_id=0
# 加载数据进程数
workers=8

device_id_list=0,1,2,3,4,5,6,7
export RANK_SIZE=8

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
if [ ! -d ${test_path_dir}/output ];then
    mkdir -p ${test_path_dir}/output
fi


#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

echo "=============start training==================="

python3 -u train8p.py \
	--addr=$(hostname -I |awk '{print $1}') \
	--seed=49  \
	--workers=${workers} \
	--train_data_path=${data_path}/VOC2012/train \
    --val_data_path=${data_path}/VOC2012/val \
    --output_dir=${test_path_dir}/output \
	--dist_url='tcp://127.0.0.1:50000' \
	--dist_backend='hccl' \
	--multiprocessing_distributed \
	--world_size=1 \
	--batch_size=${batch_size} \
	--epochs=${train_epochs} \
	--device_num=8 \
	--rank=0 \
	--amp \
	--amp_level='O1' \
    --loss_scale_g=128 \
	--loss_scale_d=128 \
	--performance=True \
	--device_list=${device_id_list}  > ${test_path_dir}/output/train_performance_8p.log

wait

##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))
# 输出性能FPS，需要模型审视修改
fps=`grep -a 'Fps:'  ${test_path_dir}/output/train_performance_8p.log|awk -F " " '{print $NF}'|tail -n 10 |awk '{a+=$1} END {if (NR != 0) printf("%.3f",a/NR)}'`
FPS=${fps%%[*}
#结果打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"
echo "Final Performance images/sec : $FPS"
