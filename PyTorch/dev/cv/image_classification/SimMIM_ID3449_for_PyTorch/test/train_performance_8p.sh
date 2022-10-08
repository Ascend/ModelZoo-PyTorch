#!/bin/bash

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1

#当前路径,不需要修改
cur_path=`pwd`
export RANK_SIZE=1
RANK_SIZE=8
echo "Device ID: $ASCEND_DEVICE_ID"

#网络名称，同目录名称
Network="SiMMIM_ID3449_for_PyTorch"
batch_size=32

# 数据集路径,保持为空,不需要修改
data_path=""
#参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

#训练开始时间，不需要修改
start_time=$(date +%s)
output_dir=$cur_path/output
#rm $output_dir -rf
#rm $cur_path/kernel_meta* -rf
#rm $cur_path/cache* -rf
mkdir $output_dir/${ASCEND_DEVICE_ID} -p

python3 -m torch.distributed.launch \
    --nproc_per_node $RANK_SIZE \
    $cur_path/../main_simmim.py \
    --cfg $cur_path/../configs/swin_base__5ep/simmim_pretrain__swin_base__img192_window6__5ep.yaml \
    --batch-size $batch_size \
    --data-path $data_path/train > $output_dir/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
EpochTime=`grep "4 training takes" $output_dir/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log |awk -F " " '{print $10}'|sed s/[[:space:]]//g`
hour=${EpochTime%%:*}
tmp=${EpochTime#*:}
minute=${tmp%%:*}
minute=${minute#*0}
second=${EpochTime##*:}
EpochSec=`python3 -c "print($hour*3600+$minute*60+$second)"`
StepPerEpoch=325

FPS=`python3 -c "print(${batch_size}*${StepPerEpoch}/${EpochSec})"`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
train_accuracy=`grep "f1:.*loss:" $output_dir/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END{print $15}'`
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`

#从train_${ASCEND_DEVICE_ID}.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep -E "eta" $output_dir/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F 'loss' '{print $2}' | sed 's/[[:space:]]\].*//g' | awk '{print $1}' > $output_dir/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print $1}' $output_dir/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
