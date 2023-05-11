data_path=""
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#################PEM Train and Test#########################
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
    python3 -u main_8p.py --module TEM --mode train --tem_batch_size 128 --tem_epoch 2 --data_path ${data_path} > ${test_path_dir}/output/0/train_perfomance_8p_${i}.log 2>&1 &

    let rank++
done

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
    taskset -c $PID_START-$PID_END python3 -u main_8p.py --module PEM --mode train --pem_batch_size 128 --pem_epoch 2  >> ${test_path_dir}/output/0/train_perfomance_8p_${i}.log 2>&1 &
    let rank++
done

wait

echo "------------------ Final result ------------------"
FPS_TEM=`grep -m 1 'FPS(TEM)'  ${test_path_dir}/output/0/train_perfomance_8p_0.log | awk -F " " '{print$4}'`
FPS_PEM=`grep -m 1 'FPS(PEM)'  ${test_path_dir}/output/0/train_perfomance_8p_0.log | awk -F " " '{print$4}'`
echo "Final TEM Performance images/sec : $FPS_TEM"
echo "Final PEM Performance images/sec : $FPS_PEM"
