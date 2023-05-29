#!/usr/bin/env bashF

data_path_info=$1
data_path=`echo ${data_path_info#*=}`
if [[ $data_path == "" ]];then
    echo "[Warning] para \"data_path\" not set"
    echo "[Warning] use default data_path"
    data_path="../data/Market-1501/"
    # exit 1
fi

RANK_ID_START=0
RANK_SIZE=1

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * RANK_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))
nohup taskset -c $PID_START-$PID_END python3 \
    PCB_amp.py --local_rank $RANK_ID --npu --device_num ${RANK_SIZE} -d market -a resnet50 -b 64 -j 4 --epochs 1 --lr 0.1 --log logs/market-1501/PCB/ --combine-trainval --feature 256 --height 384 --width 128 --step-size 40 \
    --data-dir ${data_path} --performance &
done
