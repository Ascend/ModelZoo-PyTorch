#!/bin/bash


#集合通信参数,不需要修改
export RANK_SIZE=1

# 数据集路径,保持为空,不需要修改
data_path=""

#网络名称,同目录名称,需要模型审视修改
Network="CenterNet_ID4117_for_PyTorch"
#epoch
num_epochs=1
#训练batch_size,,需要模型审视修改
batch_size=32
# 端口
port=23456
# 指定训练所使用的npu device卡id
device_id=0

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"

# 调试
bin_model=0  # 0 nobin other bin
profiling=''
start_step=-1
# stop_step=-1
num_iters=-1
# WEIGHTS
local_weights_path=resnet18-5c106cde.pth

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --num_epochs* ]];then
        num_epochs=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --port* ]];then
        port=`echo ${para#*=}`
    elif [[ $para == --bin_model* ]];then
        bin_model=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
    elif [[ $para == --start_step* ]];then
        start_step=`echo ${para#*=}`
    elif [[ $para == --num_iters* ]];then
        num_iters=`echo ${para#*=}`
    elif [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    fi
done

# 端口
cur_port=`awk 'BEGIN{printf "%d", '${port}'+'${ASCEND_DEVICE_ID}'}'`

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
# GE 模式profiling添加环境变量
if [[ $profiling == "GE" ]];then
    export GE_PROFILING_TO_STD_OUT=1
fi
# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
if [ $ASCEND_DEVICE_ID ];then
    device_id=${ASCEND_DEVICE_ID}
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

#创建DeviceID输出目录，不需要修改
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/$ASCEND_DEVICE_ID
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

# 添加二进制代码
# line=`grep "import apex" ${cur_path}/src/main_npu_8p.py -n | tail -1 | awk -F ':' '{print $1}'`
# sed -i "$[line+1]itorch.npu.set_compile_mode(jit_compile=False)" ${cur_path}/src/main_npu_8p.py

#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
# check_etp_flag=`env | grep etp_running_flag`
# etp_flag=`echo ${check_etp_flag#*=}`
# if [ x"${etp_flag}" != x"true" ];then
#     source ${test_path_dir}/env_npu.sh
# fi
#数据集处理
ln -nsf ${data_path} $cur_path/data

#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
# cd $cur_path/src
# {
# python main_npu_1p.py ctdet --exp_id pascal_resdcn18_384 --arch resdcn_18 --dataset pascal --num_epochs 5 
# } > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    
cd $cur_path/src

RANK_ID_START=0
RANK_SIZE=1
KERNEL_NUM=$(($(nproc)/8))

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
PID_START=$((KERNEL_NUM * device_id))
PID_END=$((PID_START + KERNEL_NUM - 1))
taskset -c $PID_START-$PID_END python3  main_npu_8p.py ctdet \
            --exp_id pascal_resdcn18_384 \
            --arch resdcn_18 \
            --device_list=$device_id \
            --dataset pascal \
            --num_epochs $num_epochs \
            --lr_step 45,60,75 \
            --port=$cur_port \
            --world_size 1  \
            --precision_mode=$precision_mode \
            --batch_size $batch_size \
            --num_workers ${KERNEL_NUM} \
            --bin_model ${bin_model} \
            --profiling "${profiling}" \
            --start_step ${start_step} \
            --load_local_weights True \
            --local_weights_path $cur_path/data/voc/$local_weights_path \
            --num_iters ${num_iters} \
            --local_rank $RANK_ID > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done

wait

##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a 'FPS'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log| awk -F " = " '{print $NF}'| awk 'BEGIN{count=0}{if(NR>0){sum+=$NF;count+=1}}END{printf "%.4f\n", sum/count}'`

#输出CompileTime
CompileTime=`grep 'iter_time' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|head -n 2|awk -F "iter_time = " '{print $2}'| awk '{print $1}'|awk '{sum += $1} END {print sum}'`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

#输出训练精度,需要模型审视修改
train_accuracy=$(grep metric: ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F ":" '{print $3}' | awk -F " " 'END{print $1}')
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
if [[ $precision_mode == "must_keep_origin_dtype" ]];then
        CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'fp32'_'perf'
else
        CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'
fi

#获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep metric: ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F ":" '{print $3}' | awk -F " " '{print $1}' > ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
# 最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >>$cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CompileTime = ${CompileTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
