#!/bin/bash

##################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="ResNeXt101_ID1717_for_PyTorch"
# 训练batch_size
batch_size=8192
# 训练使用的npu卡数
export RANK_SIZE=64
# 训练epoch
train_epochs=3
# 学习率
learning_rate=0.8
# 加载数据进程数
workers=128
# 数据集路径,保持为空,不需要修改
data_path=""
conf_path=""
server_index=""
fix_node_ip=""
devicesnum=""
one_node_ip=""
linux_num=""

# 参数校验，data_path为必传参数， 其他参数的增删由模型自身决定；此处若新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --workers* ]];then
        workers=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --fix_node_ip* ]];then
	    fix_node_ip=`echo ${para#*=}`
	elif [[ $para == --devicesnum* ]];then
	    devicesnum=`echo ${para#*=}`
    elif [[ $para == --conf_path* ]];then
        conf_path=`echo ${para#*=}`
    elif [[ $para == --server_index* ]];then
        server_index=`echo ${para#*=}`
    elif [[ $para == --one_node_ip* ]];then
        one_node_ip=`echo ${para#*=}`
    elif [[ $para == --linux_num* ]];then
        linux_num=`echo ${para#*=}`
    fi
done

if [[ $conf_path == "" ]];then
    one_node_ip=$one_node_ip
    linux_num=$linux_num
else 
    one_node_ip=`find $conf_path -name "server_*0.info"|awk -F "server_" '{print $2}'|awk -F "_" '{print $1}'`
    linux_num=`find $conf_path -name "server_*.info" |wc -l`
fi

PREC=""
if [[ $precision_mode == "amp" ]];then
  PREC="--amp"
fi

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi



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

export HCCL_IF_IP=$fix_node_ip
export MASTER_ADDR=$one_node_ip
export MASTER_PORT=29688
export HCCL_WHITELIST_DISABLE=1
device_num=${#devicesnum}
devices_num=`awk 'BEGIN{printf "%.0f\n",'${device_num}'-1}'`

NPUS=($(seq 0 $devices_num))
rank_server=`awk 'BEGIN{printf "%.0f\n",'${device_num}'*'${server_index}'}'`
export NPU_WORLD_SIZE=`awk 'BEGIN{printf "%.0f\n",'${device_num}'*'${linux_num}'}'`


##################启动训练脚本##################
# 训练开始时间，不需要修改
start_time=$(date +%s)
# source 环境变量
# source ${test_path_dir}/env.sh
python3.7 ./main.py \
      ${data_path} \
      --addr=$one_node_ip \
      --seed=49 \
      --workers=${workers} \
      --learning-rate=${learning_rate} \
      --mom=0.9 \
      --weight-decay=1.0e-04  \
      --print-freq=1 \
      --dist-url='tcp://127.0.0.1:23456' \
      --dist-backend 'hccl' \
      --multiprocessing-distributed \
      --world-size=${linux_num} \
      --rank=${server_index} \
      --device='npu' \
      --epochs=${train_epochs} \
      ${PREC} \
      --batch-size=${batch_size} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait



##################获取训练数据##################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
#FPS=`grep -a 'FPS'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $NF}'|awk 'END {print}'`
#打印，不需要修改
#echo "Final Performance images/sec : $FPS"

grep "FPS@all" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log > FPS.log
sed -i '1d' FPS.log
FPS=`grep "FPS@all" FPS.log | awk -F 'FPS@all' '{print $2}'|awk '{sum+=$1} END {print sum/NR}'`

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a '* Acc@1' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'|awk -F "Acc@1" '{print $NF}'|awk -F " " '{print $1}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
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
grep Epoch: ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep -v Test|awk -F "Loss" '{print $NF}' | awk -F " " '{print $1}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
