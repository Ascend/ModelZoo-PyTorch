#!/bin/bash

##################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="Retinaface_ID4015_for_PyTorch"
# 训练batch_size
batch_size=256
# 训练使用的npu卡数
export RANK_SIZE=8
# 训练epoch
# *************************新增****************************
train_epochs=100
# 学习率
# *************************新增****************************
learning_rate=0.01
# npu list
# *************************新增****************************
device_id_list=0,1,2,3,4,5,6,7
# 数据集路径,保持为空,不需要修改
data_path=""

# 参数校验，data_path为必传参数， 其他参数的增删由模型自身决定；此处若新增参数需在上面有定义并赋值；
for para in $*
do
# ************************删去如下两行*************************
#    if [[ $para == --workers* ]];then
#        workers=`echo ${para#*=}`
    if [[ $para == --precision_mode* ]];then
        apex_opt_level=`echo ${para#*=}`
                    if [[ $apex_opt_level != "O0" ]] && [[ $apex_opt_level != "O1" ]] && [[ $apex_opt_level != "O2" ]] && [[ $apex_opt_level != "O3" ]]; then
                            echo "[ERROR] para \"precision_mode\" must be config O0 O1 or O2 or O3"
                            exit 1
                    fi
        PREC="--opt-level "$apex_opt_level
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --fp32* ]];then
        fp32=`echo ${para#*=}`
    elif [[ $para == --hf32* ]];then
        hf32=`echo ${para#*=}`
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



##################启动训练脚本##################
# 训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

# ******************从D:\Pytorch_Work_Space_2\Retinaface\scripts\run_8p.sh中粘取的指令****************

python3 train.py \
    --data=${data_path}/train/label.txt \
    --addr=$(hostname -I |awk '{print $1}') \
    --workers=64 \
    --dist-url='tcp://127.0.0.1:50003' \
    --dist-backend='hccl' \
    --world-size=1 \
    --batch-size=${batch_size} \
    --lr=${learning_rate} \
    --epochs=${train_epochs} \
    --device_num=8 \
    --rank=0 \
    $prec \
    --distributed \
    $fp32 $hf32 \
    --device-list=${device_id_list} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log &
wait

##################获取训练数据##################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

weights=`find -name 'Resnet50*' | xargs ls -t | head -1`
python3 test_widerface.py -m $weights --dataset_folder ${data_path}/val/images/ &


wait

echo "Finish test_widerface.py"

cd widerface_evaluate
python3 evaluation.py --gt ${data_path}/wider_face_split/  > eval_result.txt &

wait

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
# *********************************修改了路径*********************************
FPS=`grep -a 'FPS'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $7}'|awk 'END {print}'`
FPS=`echo "${FPS} * ${RANK_SIZE}" | bc`
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

# 输出训练精度,需要模型审视修改
# ***********************************修改了路径*******************************
# 打印，不需要修改
cat eval_result.txt >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
if [[ ${fp32} == "--fp32" ]];then
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'fp32'_'acc'
elif [[ ${hf32} == "--hf32" ]];then
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'hf32'_'acc'
else
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'
fi

# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
# ********************************************修改了路径*****************************
grep Epoch: ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|grep -v Test|awk -F "Loss" '{print $NF}' | awk -F " " '{print $1}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

train_accuracy=`grep Easy ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}.log | awk '{print $NF}'`
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
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
