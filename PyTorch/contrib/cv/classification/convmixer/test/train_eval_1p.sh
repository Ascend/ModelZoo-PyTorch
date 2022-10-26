#!/bin/bash

##################基础配置参数，需要模型审视修改##################
# 指定参数 --data_path=XXX
# 网络名称，同目录名称
Network="convmixer_1536_20"
# 所选模型
model="convmixer_1536_20"
# 训练batch_size
batch_size=64
# 训练使用的npu卡数
RANK_SIZE=1
# 数据集类别数量
nb_classes=1000
# 数据集路径,保持为空,不需要修改
data_path=""
# 权重文件路径
checkpoint=""

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_eval_1P.sh <args>"
    echo " "
    echo "parameter explain:
    --model                    choose the training model  
    --checkpoint                           
    --nb_classes               numbers of data classes
    --data_path		           source val data
    -h/--help		           show help message
    "
    exit 1
fi

#参数校验，不需要修改                   ***********************************?
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --nb_classes* ]];then
        nb_classes=`echo ${para#*=}`
    elif [[ $para == --checkpoint* ]];then
        checkpoint=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
if [[ $checkpoint == "" ]];then
    echo "[Error] para \"checkpoint\" must be confing"
    exit 1
fi

echo "data_path: $data_path"
echo "checkpoint: $checkpoint"

##################指定训练脚本执行路径##################            
# cd到与test文件同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi
echo "cur_path: ${cur_path}"
##################创建日志输出目录，根据模型审视################## 
# 模型采用非循环方式启动多卡训练，创建日志输出目录如下；采用循环方式启动多卡训练的模型，在循环中创建日志输出目录，可参考CRNN模型
# 非循环方式下8卡训练日志输出路径中的ASCEND_DEVICE_ID默认为0，只是人为指定文件夹名称， 不涉及训练业务
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/$ASCEND_DEVICE_ID ];then
    rm -rf ${test_path_dir}/output/$ASCEND_DEVICE_ID
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi
echo "test_path_dir: ${test_path_dir}"
##################启动训练脚本##################
#训练开始时间，不需要修改
start_time=$(date +%s)
# source 环境变量
source ./test/env_npu.sh
python3 validate_npu.py \
--model ${model} \
--b ${batch_size} \
--num-classes ${nb_classes} \
--checkpoint ${checkpoint} \
${data_path} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/eval_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

##################获取训练数据##################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

# 输出测试精度,需要模型审视修改
# Test: [  97/97]  Time: 0.659 (0.298)  Loss:  0.9513 (1.0809)  Acc@1: 80.0595 (80.2100)  Acc@5: 97.0238 (95.1520)
eval_accuracy_acc1=`grep -a 'Test: ' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/eval_${ASCEND_DEVICE_ID}.log|awk -F "(" '{print $4}'|awk -F ")" '{print $1}'|awk 'END {print}'`
eval_accuracy_acc5=`grep -a 'Test: ' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/eval_${ASCEND_DEVICE_ID}.log|awk -F "(" '{print $5}'|awk -F ")" '{print $1}'|awk 'END {print}'`
# 打印，不需要修改
echo "Final Train Accuracy : Acc@1: ${eval_accuracy_acc1} % ,  Acc@5: ${eval_accuracy_acc5}"
echo "E2E Training Duration sec : $e2e_time"

