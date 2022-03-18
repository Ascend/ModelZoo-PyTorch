#!/bin/bash
source scripts/env_new.sh
export ETCD_UNSUPPORTED_ARCH=arm64
echo "train log path is 8p_train_performance.log"
{
startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`
echo "start time: $startTime"

python3.7 -m torchelastic.distributed.launch \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  train.py \
  data.train_manifest=data/an4_train_manifest.csv \
  data.val_manifest=data/an4_val_manifest.csv \
  data.num_workers=32 \
  apex.opt_level=O2 \
  apex.loss_scale=1 \
  data.batch_size=10 \
  training.epochs=3 \
  optim=adam \
  optim.learning_rate=8e-4 \
  optim.learning_anneal=0.99
wait
 
endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
echo "end time: $endTime"
sumTime_sec=$(( $endTime_s - $startTime_s ))
sumTime_min=$(( $sumTime_sec / 60 ))
echo "$startTime ---> $endTime" "Total: $sumTime_min minutes"
fps=$(echo $(printf "%.3f" `echo "scale=3; 808 * 8 * 3 / $sumTime_sec"|bc`))
echo "FPS -----> $fps"
} > ./8p_train_performance.log 2>&1 & 
