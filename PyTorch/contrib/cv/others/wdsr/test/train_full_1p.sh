#!/bin/bash

# 指定训练所使用的npu device卡id
device_id=0
# 参数校验
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    fi
done

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
source ${test_path_dir}/env_npu.sh
nohup taskset -c 0-23 python3 -u trainer.py \
                --dataset div2k \
                --eval_datasets div2k \
                --model wdsr \
                --scale 2 \
                --local_rank ${ASCEND_DEVICE_ID} \
                --job_dir ./wdsr_x2 &
