#!/bin/bash
##################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="Retinaface_ID4015_for_PyTorch"
# 训练batch_size
batch_size=32
# 训练使用的npu卡数
export RANK_SIZE=1
# 数据集路径,保持为空,不需要修改
data_path=""

# 训练epoch
train_epochs=1
# 指定训练所使用的npu device卡id
device_id=0
# 学习率
learning_rate=0.001


# 参数校验，data_path为必传参数， 其他参数的增删由模型自身决定；此处若新增参数需在上面有定义并赋值

for para in $*
do
    if [[ $para == --precision_mode* ]];then
        apex_opt_level=`echo ${para#*=}`
                    if [[ $apex_opt_level != "O0" ]] && [[ $apex_opt_level != "O1" ]] && [[ $apex_opt_level != "O2" ]] && [[ $apex_opt_level != "O3" ]]; then
                            echo "[ERROR] para \"precision_mode\" must be config O0 O1 or O2 or O3"
                            exit 1
                    fi
        PREC="--opt-level "$apex_opt_level
    elif [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --fp32* ]];then
        fp32=`echo ${para#*=}`
    elif [[ $para == --hf32* ]];then
        hf32=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

if [[ $apex_opt_level == "O0" ]];then
    prec=$PREC
else
    prec="--loss-scale=128. --amp $PREC"
fi   
# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

# 校验单卡训练是否指定了device id，分动态分配device id 与手动指定device id，此处不需要修改
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ]; then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    echo "[Error] device id must be confing"
    exit 1
fi


#################指定训练脚本执行路径##################
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


##################创建日志输出目录，不需要修改##################

if [ -d ${test_path_dir}/output/$ASCEND_DEVICE_ID ];then
    rm -rf ${test_path_dir}/output/$ASCEND_DEVICE_ID
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi
##################启动训练脚本##################
# 训练开始时间，不需要修改
start_time=$(date +%s)
python3 train.py \
    --data=${data_path}/train/label.txt \
    --addr=$(hostname -I |awk '{print $1}') \
    --workers=64 \
    --dist-url='tcp://127.0.0.1:50000' \
    --dist-backend='hccl' \
    --world-size=1 \
    --batch-size=${batch_size} \
    --epochs=${train_epochs} \
    --rank=0 \
    $prec \
    --max_steps=100 \
    $fp32 $hf32 \
    --device-list=${ASCEND_DEVICE_ID} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log &
wait

##################获取训练数据##################
# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 终端结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
# ***********************修改了路径***************************
FPS=`grep -a 'FPS'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $7}'|awk 'END {print}'`
CompileTime=`grep -a 'Epoch'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|grep Time|head -n 2|awk -F "Time" '{sum+=$2}END{print sum}'`
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

# 输出训练精度,需要模型审视修改
# ***********************修改了路径***************************
train_accuracy=`grep -a '* Acc@1' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print}'|awk -F "Acc@1" '{print $NF}'|awk -F " " '{print $1}'`
# 打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
if [[ ${fp32} == "--fp32" ]];then
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'fp32'_'perf'
elif [[ ${hf32} == "--hf32" ]];then
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'hf32'_'perf'
else
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'
fi
# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
# ***********************修改了路径***************************
grep Epoch: ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|grep -v Test|awk -F "Loss" '{print $NF}' | awk -F " " '{print $1}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

# 最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`


##################将训练数据存入文件##################
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
echo "CompileTime = ${CompileTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
