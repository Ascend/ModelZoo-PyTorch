#!/usr/bin/env python

source test/env_npu.sh

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

#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)

RANK_ID_START=0
RANK_SIZE=8

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do

KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))

nohup taskset -c $PID_START-$PID_END python3 -u eval/eval_iou.py \
--datadir ${data_path} \
--loadDir "save/erfnet_training1/" \
--loadWeights "model_best.pth" \
--amp \
--opt-level "O2" \
--loss-scale-value "dynamic" \
--device npu \
--num_gpus 8 \
--num-workers 32 \
--batch-size 24 \
--rank_id $RANK_ID > erfnet_eval.log 2>&1 &
done

wait

##################获取训练数据################
# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 结果打印，不需要修改
echo "------------------ Final result ------------------"

# 输出训练精度,需要模型审视修改
iou=`grep 'MEAN IoU' erfnet_eval.log |tail -n 1|awk '{print $3}'|awk 'END {print}'`
# 打印，不需要修改
echo "Final Train IoU: ${iou}"
echo "E2E Training Duration sec : $e2e_time"