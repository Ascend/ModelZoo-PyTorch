#!/bin/bash

#集合通信参数,不需要修改
export RANK_SIZE=1
# 数据集路径,保持为空,不需要修改
device_id=0

#基础参数，需要模型审视修改
#网络名称，同目录名称
#训练epoch
train_epochs=3
#训练任务
TASK=""
#训练batch_size 默认bert base batch size, 该参数外部可传入
batch_size=32

#获取外部传参，可扩展
for para in $*
do
    if [[ $para == --train_epochs* ]];then
      train_epochs=`echo ${para#*=}`
    elif [[ $para == --TASK* ]];then
      TASK=`echo ${para#*=}`
    fi
done

#校验是否传入参数,不需要修改
if [[ $train_epochs == "" ]];then
    echo "[Error] para \"train_epochs\" must be confing"
    exit 1
elif [[ $TASK == "" ]];then
    echo "[Error] para \"TASK\" must be confing"
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

if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi

#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

if [ $(uname -m) = "aarch64" ]
    then
        nohup taskset -c 0-23 python3 run_glue.py --model_name_or_path bert-large-cased \
                --task_name $TASK \
                --do_train \
                --device ${device_id} \
                --do_eval \
                --dataloader_num_workers $(($(nproc)/8)) \
                --max_seq_length 128 \
                --per_device_train_batch_size 32 \
                --learning_rate 2e-5 \
                --num_train_epochs $train_epochs \
                --fp16 \
                --fp16_opt_level O2 \
                --dataloader_drop_last \
                --loss_scale 1024.0 \
                --optim adamw_apex_fused_npu \
                --use_combine_grad \
                --overwrite_output_dir \
                --save_steps 50000\
                --skip_steps 5 \
                --output_dir /tmp/$TASK/ > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    else
        nohup  python3 run_glue.py --model_name_or_path bert-large-cased \
                --task_name $TASK \
                --do_train \
                --device ${device_id} \
                --do_eval \
                --dataloader_num_workers $(($(nproc)/8)) \
                --max_seq_length 128 \
                --per_device_train_batch_size 32 \
                --learning_rate 2e-5 \
                --num_train_epochs $train_epochs \
                --fp16 \
                --fp16_opt_level O2 \
                --dataloader_drop_last \
                --loss_scale 1024.0 \
                --optim adamw_apex_fused_npu \
                --use_combine_grad \
                --overwrite_output_dir \
                --save_steps 50000\
                --skip_steps 5 \
                --output_dir /tmp/$TASK/ > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    fi
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep "train_samples_per_second ="  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $3}'`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
if [ x"${TASK}" == x"cola" ];then
    train_accuracy=`grep "eval_matthews_correlation" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $3}'`
elif [ x"${TASK}" == x"stsb" ];then
    train_accuracy=`grep "eval_spearmanr" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $3}'`
elif [ x"${TASK}" == x"mrpc" ] || [ x"${TASK}" == x"qqp" ];then
    train_accuracy=`grep "eval_f1" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $3}'`
elif [ x"${TASK}" == x"mnli" ];then
     train_accuracy_match=`grep "eval_accuracy" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log |head -n 1 |awk 'END {print $3}'`
     train_accuracy_mismatch=`grep "eval_accuracy" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $3}'`
     train_accuracy=${train_accuracy_match}' '${train_accuracy_mismatch}
else
    train_accuracy=`grep "eval_accuracy" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $3}'`
fi

echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "{'loss" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log |awk -F "{'loss" '{print $2}'|awk '{print $2}' | awk -F "," '{print $1}' > ${test_path_dir}/output/${ASCEND_DEVICE_ID}//train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}">> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log