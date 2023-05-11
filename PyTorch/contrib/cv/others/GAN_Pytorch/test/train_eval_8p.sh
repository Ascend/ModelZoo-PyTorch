#!/bin/bash
currentDir=$(cd "$(dirname "$0")";pwd)/..

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
#export RANK_SIZE=8

# 数据集路径,保持为空,不需要修改
data_path=""

#网络名称,同目录名称,需要模型审视修改
Network="GAN"

#训练batch_size,,需要模型审视修改
batch_size=64



#参数校验，不需要修改
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
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
    export ASCEND_DEVICE_ID=0,1,2,3,4,5,6,7
    echo "device id is ${ASCEND_DEVICE_ID}"
fi

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

#进入训练脚本目录，需要模型审视修改
cd $cur_path/

#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/output/${Network}/${ASCEND_DEVICE_ID} ];then
    rm -rf ${cur_path}/output/${Network}/${ASCEND_DEVICE_ID}
    mkdir -p ${cur_path}/output/${Network}/$ASCEND_DEVICE_ID/ckpt
else
    mkdir -p ${cur_path}/output/${Network}/$ASCEND_DEVICE_ID/ckpt
fi

#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${cur_path}/test/env_npu.sh
fi

#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
python3 -u -m torch.distributed.launch --nproc_per_node=8 ${currentDir}/main.py \
        --distributed \
        --lr 0.0008 \
        --batch_size 128 \
        --n_epochs 200 \
        --workers 0 \
        --apex \
        --test_only 1 \
        --data_path ${data_path} > ${cur_path}/output/train_eval_8p.log 2>&1 &

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
echo "end_time: ${end_time}"
e2e_time=$(( $end_time - $start_time ))


