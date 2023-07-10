##################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 删除之前结果
rm -rf ./results

#集合通信参数,不需要修改
export RANK_SIZE=8

# 数据集路径,保持为空,不需要修改
data_folder=""
#网络名称,同目录名称,需要模型审视修改
Network="Tdnn"

#训练batch_size,,需要模型审视修改
batch_size=64

#训练epoch
train_epochs=35
# 指定训练所使用的npu device卡id
device_id=0


# 参数校验，data_folder为必传参数， 其他参数的增删由模型自身决定；此处若新增参数需在上面有定义并赋值；
for para in $*
do
    if [[ $para == --workers* ]];then
        workers=`echo ${para#*=}`
    elif [[ $para == --data_folder* ]];then
        data_folder=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    fi
done

# 校验是否传入data_folder,不需要修改
if [[ $data_folder == "" ]];then
    echo "[Error] para \"data_folder\" must be config"
    exit 1
fi

##################指定训练脚本执行路径##################
# cd到与test文件同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
    ln -s  source  dest
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
ASCEND_DEVICE_ID=${device_id}
if [ -d ${test_path_dir}/output/$ASCEND_DEVICE_ID ];then
    rm -rf ${test_path_dir}/output/$ASCEND_DEVICE_ID
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

##################启动训练脚本##################
# 训练开始时间，不需要修改
start_time=$(date +%s)
# source 环境变量
source ${test_path_dir}/env_npu.sh
export WORLD_SIZE=8
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='11225'
KERNEL_NUM=$(($(nproc)/8))
for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
do
    export RANK=$RANK_ID
    PID_START=$((KERNEL_NUM * RANK_ID))
    PID_END=$((PID_START + KERNEL_NUM - 1))
    nohup taskset -c $PID_START-$PID_END python3 train.py train.yaml \
			--distributed_launch \
			--distributed_backend=hccl \
			--local_rank ${RANK_ID} \
			--batch_size=$batch_size \
			--number_of_epochs=$train_epochs \
			--data_folder=$data_folder >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}_perf_8p.log 2>&1 &

done
wait


##################获取训练数据##################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
FPS=`grep -a 'FPS'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}_perf_8p.log|awk -F "(" '{print $NF}'|awk 'END {print}'|cut -d ")" -f1`
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "npu id:" ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${ASCEND_DEVICE_ID}_perf_8p.log|awk -F "Loss" '{print $NF}' | awk -F " " '{print $1}' >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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

