#!/bin/bash

cur_path=`pwd`
nmon -s3 -c 500 -f -m $cur_path
#集合通信参数,不需要修改
export RANK_SIZE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29688
export HCCL_WHITELIST_DISABLE=1
# 数据集路径,保持为空,不需要修改
data_path=""

#网络名称,同目录名称,需要模型审视修改
Network="Transformer_ID0105_for_PyTorch"

#训练batch_size,,需要模型审视修改
batch_size=128



# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --epochs* ]];then
        epochs=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi




#创建DeviceID输出目录，不需要修改
if [ -d $cur_path/output/${ASCEND_DEVICE_ID} ];then
    rm -rf $cur_path/output/$ASCEND_DEVICE_ID
    mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
else
    mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
fi

#################启动训练脚本#################


# 必要参数替换配置文件
cd $cur_path/..
DATA_DIR=./data/dataset/wmt14_en_de_joined_dict/
MODELDIR="./checkpoints/"
mkdir -p "$MODELDIR"
LOGFILE="$MODELDIR/log"
STAT_FILE="log.txt"

sed -i "s|if i>100:pass|if i>100:break|g" train_8p_new.py
sed -i "s|if m >=2:pass|if m >=2:break|g" train_8p_new.py

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL_ETP=3
export PTCOPY_ENABLE=1
export TASK_QUEUE_ENABLE=1
export DYNAMIC_OP="ADD#MUL"
start_time=$(date +%s)
NPUS=($(seq 0 7))
export NPU_WORLD_SIZE=${#NPUS[@]}
rank=0
for i in ${NPUS[@]}
do
    export NPU_CALCULATE_DEVICE=${i}
    mkdir -p  $cur_path/output/${i}/
    export ASCEND_DEVICE_ID=${i}
    export RANK=${rank}
    echo run process ${rank}


    python3 train_8p_new.py \
       $data_path \
      --arch transformer_wmt_en_de \
      --share-all-embeddings \
      --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.997 \
  --addr '127.0.0.1' \
  --port 29990 \
  --adam-eps "1e-9" \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 0.0 \
  --warmup-updates 4000 \
  --lr 0.0006 \
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-sentences 128\
  --max-tokens 102400 \
  --seed 1 \
  --save-dir $MODELDIR \
  --stat-file $STAT_FILE\
  --log-interval 1\
  --amp\
  --device-id ${rank}\
  --amp-level O2  >  $cur_path/output/${i}/train_${i}.log 2>&1 &
    let rank++
done
wait
sed -i "s|if i>100:break|if i>100:pass|g" train_8p_new.py
sed -i "s|if m >=2:break|if m >=2:pass|g" train_8p_new.py 
    


##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=` grep -rns "Time" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log |grep -v "all" |awk -F "Time" '{print$2}' |awk -F "(" '{print$1}'|tail -n +5|awk '{sum+=$1} END {print"",8*128*NR/sum}'|sed s/[[:space:]]//g`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

#获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep -rns "Time" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log |grep -v "all" |awk -F "Loss" '{print$2}' |awk -F "(" '{print$1}'  > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${cur_path}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${cur_path}/output/$ASCEND_DEVICE_ID/${CaseName}.log