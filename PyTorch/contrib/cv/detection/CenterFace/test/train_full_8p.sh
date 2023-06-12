#!/bin/bash


#集合通信参数,不需要修改
export RANK_SIZE=8


# 数据集路径,保持为空,不需要修改
data_path=""

#网络名称,同目录名称,需要模型审视修改
Network="CenterFace_ID4089_for_PyTorch"

#训练batch_size,,需要模型审视修改
batch_size=16

#参数校验，不需要修改
# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --hf32 ]];then
        hf32=`echo ${para#*=}`
    elif [[ $para == --fp32 ]];then
        fp32=`echo ${para#*=}`
    elif [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    fi
done

if [[ $precision_mode == "must_keep_origin_dtype" ]];then
    PREC=" --use_fp32 "
else
    PREC=""
fi


#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
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

#创建DeviceID输出目录，不需要修改
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/$ASCEND_DEVICE_ID
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
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

# 数据集软链到脚本工程内部
cur_path=`pwd`
default_data_path=${cur_path}/data/
mkdir -p ${cur_path}/exp/multi_pose/mobilev2_10/debug/
mkdir -p ${cur_path}/output/widerface/
mkdir -p ${cur_path}/data/
rm -rf ${default_data_path}/wider_face
ln -s ${data_path} ${default_data_path}/.

# 训练前模型编译
cd $cur_path/src/lib/external
python3 setup.py build_ext --inplace
wait

#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
cd $cur_path/src
RANK_ID_START=0
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++))
do
    echo ${RANK_ID}
    KERNEL_NUM=$(($(nproc)/8))
    PID_START=$((KERNEL_NUM * RANK_ID))
    PID_END=$((PID_START + KERNEL_NUM - 1))
    taskset -c $PID_START-$PID_END python3  main.py $PREC \
    --device_list='0,1,2,3,4,5,6,7' \
    --world_size=8 \
    --batch_size=$batch_size \
    --local_rank ${RANK_ID} \
    --lr=2.5e-3 \
    --lr_step='85,120' \
    --num_epochs=160 \
    ${fp32} \
    ${hf32} \
    --distributed_launch > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done
wait

python3 test_wider_face.py >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

cd $cur_path/evaluate
python3 setup.py build_ext
python3 evaluation.py --pred $cur_path/output/widerface >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a 'FPS'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F " = " '{print $NF}'|awk 'NR==1{max=$1;next}{max=max>$1?max:$1}END{print max}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a 'Easy' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $NF}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`

if [[ ${fp32} == "--fp32" ]];then
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'fp32'_'acc'
elif [[ ${hf32} == "--hf32" ]];then
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'hf32'_'acc'
else
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'
fi

#获取性能数据，不需要修改
#吞吐量
ActualFPS=`awk -v x="$FPS" -v y="$RANK_SIZE" 'BEGIN{printf "%.3f\n", x*y}'`
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
# grep Epoch: ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "Loss " '{print $NF}' | awk -F " " '{print $1}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
# ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
# echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
