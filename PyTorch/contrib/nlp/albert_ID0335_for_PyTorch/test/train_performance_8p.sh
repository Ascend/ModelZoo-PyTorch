# encoding=utf-8

# 数据集路径,保持为空,不需要修改
data_path=""
batch_size=440
Network='albert_ID0335_for_PyTorch'
RANK_SIZE=8
# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --workers* ]];then
        workers=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_diename=${cur_path##*/}
if [ x"${cur_path_last_diename}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi
# 指定训练所使用的npu device卡id
device_id=0

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

#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source  ${test_path_dir}/env_npu.sh
fi

#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

rm -rf ./outputs
mkdir ./outputs
cp -r $data_path/prev_trained_model ./
export BERT_BASE_DIR=./prev_trained_model/albert_base_v2
export DATA_DIR=${data_path}
export OUTPUR_DIR=./outputs
export PYTHONPATH=./:$PYTHONPATH
export DEVICE=npu
TASK_NAME="SST-2"
#训练开始时间，不需要修改
start_time=$(date +%s)
KERNEL_NUM=$(($(nproc)/8))
for i in $(seq 7 -1 0)
    do
        PID_START=$((KERNEL_NUM * i))
        PID_END=$((PID_START + KERNEL_NUM - 1))
        nohup taskset -c $PID_START-$PID_END python3.7 ./run_classifier.py \
        --device=$DEVICE \
        --model_type=$BERT_MODEL \
        --model_name_or_path=$BERT_BASE_DIR/$BERT_MODEL \
        --task_name=$TASK_NAME \
        --data_dir=$DATA_DIR/$TASK_NAME/ \
        --spm_model_file=$BERT_BASE_DIR/$BERT_MODEL/30k-clean.model \
        --output_dir=$OUTPUR_DIR/$TASK_NAME/ \
        --do_train \
        --do_eval \
        --do_lower_case \
        --max_seq_length=128 \
        --batch_size=${batch_size} \
        --learning_rate=180e-5 \
        --num_train_epochs=2.0 \
        --logging_steps=10 \
        --save_steps=10 \
        --overwrite_output_dir \
        --seed=42 \
        --local_rank=$i \
        --fp16 \
        --fp16_opt_level=O2  > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    done

wait
##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a 'FPS'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "FPS = " '{print $2}'|awk 'NR==1{max=$1;next}{max=max>$1?max:$1}END{print max}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
#输出训练精度,需要模型审视修改
train_accuracy=`grep -a 'acc'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "acc = " '{print $2}'|tail -1`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

#获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep -a 'loss' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "loss: " '{print $2}' |awk -F " " '{print $1}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
