#!/usr/bin/env bash
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

nohup python3 -u train/main.py \
    --datadir ${data_path} \
    --decoder \
    --pretrainedEncoder "trained_models/erfnet_encoder_pretrained.pth.tar" \
    --num-epochs 3 \
    --amp \
    --opt-level "O2" \
    --loss-scale-value "dynamic" > erfnet_1p_perf.log 2>&1 &

wait
   
##################获取训练数据################
# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
FPS=`grep 'now epoch' erfnet_1p_perf.log|tail -n 1|awk '{print $7}'|awk 'END {print}'`
FPS=${FPS%,*}
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

# 输出训练精度,需要模型审视修改
iou=`grep 'EPOCH IoU' erfnet_1p_perf.log|tail -n 1|awk '{print $6}'|awk 'END {print}'`
# 打印，不需要修改
echo "Final Train IoU: ${iou}"
echo "E2E Training Duration sec : $e2e_time"