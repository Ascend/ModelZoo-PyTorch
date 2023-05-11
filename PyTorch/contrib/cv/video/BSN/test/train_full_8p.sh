# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值

data_path=""

for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
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
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#################创建日志输出目录，不需要修改#################
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

#################TEM Train and Test#########################
export MASTER_ADDR=localhost
export MASTER_PORT=29688
export HCCL_WHITELIST_DISABLE=1

NPUS=($(seq 0 7))
export NPU_WORLD_SIZE=${#NPUS[@]}
rank=0
for i in ${NPUS[@]}
do
    export NPU_CALCULATE_DEVICE=${i}
    export RANK=${rank}
    echo run process ${rank}
    python3 -u main_8p.py --module TEM --mode train --tem_batch_size 128 --data_path ${data_path} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_tem_${i}.log 2>&1 &

    let rank++
done

wait

python3 -u main_1p.py --module TEM --mode inference >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_tem_0.log 2>&1 &

wait

#################PEM Train and Test#########################
NPUS=($(seq 0 7))
export NPU_WORLD_SIZE=${#NPUS[@]}
rank=0
for i in ${NPUS[@]}
do
    export NPU_CALCULATE_DEVICE=${i}
    export RANK=${rank}
    echo run process ${rank}
    KERNEL_NUM=$(($(nproc)/8))
    PID_START=$((KERNEL_NUM * RANK))
    PID_END=$((PID_START + KERNEL_NUM - 1))
    taskset -c $PID_START-$PID_END python3 -u main_8p.py --module PEM --mode train --pem_batch_size 128  > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_pem_${i}.log 2>&1 &
    let rank++
done    

wait

python3 -u main_1p.py --module PEM --mode inference >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_pem_0.log 2>&1 &

wait
#################获取训练数据################
#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS_TEM=`grep -m 1 'FPS(TEM)'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_tem_${ASCEND_DEVICE_ID}.log | awk -F " " '{print$4}'`
FPS_PEM=`grep -m 1 'FPS(PEM)'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_pem_${ASCEND_DEVICE_ID}.log | awk -F " " '{print$4}'`
#打印，不需要修改
echo "Final TEM Performance images/sec : $FPS_TEM"
echo "Final PEM Performance images/sec : $FPS_PEM"
#输出训练精度,需要模型审视修改
train_accuracy=`grep -- "AR@100"  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_pem_${ASCEND_DEVICE_ID}.log | awk -F " " '{print$3}'`
#train_accuracy=`grep -- "AR@100"  ${test_path_dir}/output/0/train_pem_0.log | awk -F " " '{print$3}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
