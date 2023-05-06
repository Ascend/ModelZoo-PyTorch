#!/bin/bash
RANK_SIZE=1
#网络名称，同目录名称
Network="DynamicUNet_RT1_ID4080_for_PyTorch"
batch_size=4  # 与训练实际batch_size保持一致

# 数据集路径,保持为空,不需要修改
data_path=""
epochs=50
log_iter=1
lr=0.0001
worker=8
val_epoch=5

# 预训练模型路径
more_path1=""

# NPU调试参数
profiling="None"
start_step=0
stop_step=20
bin_mode=False
perf_iter=-1

#精度参数
precision_mode="allow_mix_precision"

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --more_path1* ]];then
        more_path1=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --epochs* ]];then
        epochs=`echo ${para#*=}`
    elif [[ $para == --log_iter* ]];then
        log_iter=`echo ${para#*=}`
    elif [[ $para == --lr* ]];then
        lr=`echo ${para#*=}`
    elif [[ $para == --worker* ]];then
        worker=`echo ${para#*=}`
    elif [[ $para == --val_epoch* ]];then
        val_epoch=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
    elif [[ $para == --start_step* ]];then
        start_step=`echo ${para#*=}`
    elif [[ $para == --stop_step* ]];then
        stop_step=`echo ${para#*=}`
    elif [[ $para == --bin_mode* ]];then
        bin_mode=`echo ${para#*=}`
    elif [[ $para == --perf_iter* ]];then
        perf_iter=`echo ${para#*=}`
    elif [[ $para == --hf32 ]];then
        hf32=`echo ${para#*=}`
    elif [[ $para == --fp32 ]];then
        fp32=`echo ${para#*=}`
    elif [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
	fi
done

if [[ $profiling == "GE" ]]; then
    export GE_PROFILING_TO_STD_OUT=1
fi

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

if [[ $more_path1 == "" ]];then
	pretrained_model="./"
else
	pretrained_model=${more_path1}/resnet50-19c8e357.pth
fi

if [[ $precision_mode == "must_keep_origin_dtype" ]];then
   prec=""
else
   prec="--amp"
fi

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
	test_path_dir=${cur_path}
	cd ..
	cur_path=$(pwd)
else
	test_path_dir=${cur_path}/test
fi

# ASCEND_DEVICE_ID=0

#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ]; then
	rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
	mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
	mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

# 使能runtime1.0。
export ENABLE_RUNTIME_V2=0

#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
# check_etp_flag=$(env | grep etp_running_flag)
# etp_flag=$(echo ${check_etp_flag#*=})
# if [ x"${etp_flag}" != x"true" ]; then
# 	source ${test_path_dir}/env_npu.sh
# fi

# train
export PYTHONPATH=./awesome-semantic-segmentation-pytorch:$PYTHONPATH

nohup python3 -u runner.py \
--model dynamicunet ${prec} \
--dataset pascal_voc --dataset-path ${data_path} \
--warmup-iters 20 \
--lr ${lr} \
--epochs ${epochs} \
--worker ${worker} \
--log-iter ${log_iter} \
--val-epoch ${val_epoch} \
--pretrained ${pretrained_model} \
--profiling ${profiling} \
--start_step ${start_step} \
--stop_step ${stop_step} \
--perf_iter ${perf_iter} \
--bin_mode ${bin_mode} \
--precision_mode ${precision_mode} \
${hf32} ${fp32} >${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=$(grep "FPS:" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F "FPS:" '{print $2}' | awk -F " " '{print $1}' | awk 'BEGIN{count=0}{if(NR>2){sum+=$NF;count+=1}}END{printf "%.4f\n", sum/count}')

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=$(grep "mIoU" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $NF}')

#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=$(uname -m)
if [[ ${fp32} == "--fp32" ]];then
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'fp32'_'acc'
elif [[ ${hf32} == "--hf32" ]];then
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'hf32'_'acc'
else
  CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'
fi

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=$(awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}')

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "FPS" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | grep "Loss:" | awk -F "Loss:" '{print $2}' |awk '{print $1}' >${test_path_dir}/output/${ASCEND_DEVICE_ID}//train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=$(awk 'END {print}' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt)

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "CaseName = ${CaseName}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >>${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
