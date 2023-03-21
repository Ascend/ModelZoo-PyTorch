#!/bin/bash


#集合通信参数,不需要修改
export RANK_SIZE=1
#设置默认日志级别,不需要修改
# export ASCEND_GLOBAL_LOG_LEVEL=3

# 数据集路径,保持为空,不需要修改
data_path=""
profiling="NONE"


#网络名称,同目录名称,需要模型审视修改
Network="CRNN_RT2_ID0103_for_PyTorch"

#训练batch_size,,需要模型审视修改
batch_size=2560

#训练epoch，不需要修改
epochs=1
start_step=700
stop_step=720
max_step=10
bin=True
pro=True 
training_debug=False 
training_type=False 

# 指定训练所使用的npu device卡id
device_id=6
# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --epochs* ]];then
        epochs=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
    elif [[ $para == --start_step* ]];then
        start_step=`echo ${para#*=}`
    elif [[ $para == --stop_step* ]];then
        stop_step=`echo ${para#*=}`
    elif [[ $para == --bin* ]];then
        bin=`echo ${para#*=}`
    elif [[ $para == --pro* ]];then
        pro=`echo ${para#*=}`
    elif [[ $para == --training_debug* ]];then
        training_debug=`echo ${para#*=}`
    elif [[ $para == --training_type* ]];then
        training_type=`echo ${para#*=}`
    elif [[ $para == --max_step* ]];then
        max_step=`echo ${para#*=}`
    elif [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --ND* ]];then
        ND=`echo ${para#*=}`
    fi
done

unset PTCOPY_ENABLE
unset SCALAR_TO_HOST_MEM
unset COMBINED_ENABLE
unset HCCL_CONNECT_TIMEOUT
unset MOTD_SHOWN
unset DYNAMIC_OP
unset TASK_QUEUE_ENABLE
unset HCCL_WHITELIST_DISABLE

if [[ $profiling == "GE" ]];then
    export GE_PROFILING_TO_STD_OUT=1
fi

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
if [ $ASCEND_DEVICE_ID ];then
    device_id=$ASCEND_DEVICE_ID
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

#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

# 必要参数替换配置文件
cur_path=`pwd`
sed -i "0,/BATCH_SIZE_PER_GPU.*$/s//BATCH_SIZE_PER_GPU\: ${batch_size}/g" ${cur_path}/LMDB_config_pr.yaml
sed -i "s/END_EPOCH.*$/END_EPOCH\: ${epochs}/g" ${cur_path}/LMDB_config_pr.yaml
sed -i "s|TRAIN_ROOT.*$|TRAIN_ROOT\: ${data_path}/MJ_LMDB|g" ${cur_path}/LMDB_config_pr.yaml
sed -i "s|TEST_ROOT.*$|TEST_ROOT\: ${data_path}/IIIT5K_lmdb|g" ${cur_path}/LMDB_config_pr.yaml

#执行训练脚本，以下传参不需要修改，其他需要模型审视修改

python3 main_8p.py \
    --bin=${bin}\
    --pro=${pro} \
    --training_debug=${training_debug} \
    --training_type=${training_type} \
    --max_step ${max_step} \
    --npu ${device_id} \
    --profiling ${profiling} \
	--start_step ${start_step} \
	--stop_step ${stop_step} \
	  --ND ${ND} \
    --cfg LMDB_config_pr.yaml > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
   
wait

##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#输出CompileTime，需要模型审视修改
compile_time=`grep "Fps" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "Time" '{print $2}' | awk -F " " '{print $1}'|head -1|awk -F 'ms' '{print $1}'`
CompileTime=`awk 'BEGIN{printf "%.2f\n", '${compile_time}'/'1000'}'`

#输出性能FPS，需要模型审视修改
TIME=`grep "Fps" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "Time" '{print $2}' | awk -F " " '{print $1}'| awk 'BEGIN{count=0}{if(NR>2){sum+=$NF;count+=1}}END{printf "%.4f\n", sum/count}'`
FPS=`awk 'BEGIN{printf "%.2f\n", '1000*$batch_size'/'${TIME}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

train_accuracy=`grep -a 'best acc is:' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END{print}'|awk -F " " '{print $NF}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
#解析GE profiling
# if [ ${profiling} == "GE" ];then
#    echo "GE profiling is loading-------------------------------------"
#    path=`find ./ -name "PROF*"`
#    ada-pa ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log
# fi

#解析CANN profiling
# if [ ${profiling} == "CANN" ];then
# 	echo "CANN profiling is loading-------------------------------------"
#     profs=`find ./ -name "PROF*" `
# 	prof_path='/usr/local/Ascend/CANN-1.84/tools/profiler/profiler_tool/analysis/msprof'
# 	iter=0
# 	for prof in ${profs}
# 	do
# 		iter=$((iter+1))
# 		python ${prof_path}/msprof.py import -dir ${prof}
# 		python ${prof_path}/msprof.py export timeline -dir ${prof} --iteration-id  $iter
# 		python ${prof_path}/msprof.py export summary -dir ${prof} --format csv --iteration-id  $iter
# 	done
# fi


#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf_nd'

#获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep -a 'Loss' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "Loss " '{print $NF}' | awk -F " " '{print $1}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`


#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >>${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CompileTime = ${CompileTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
