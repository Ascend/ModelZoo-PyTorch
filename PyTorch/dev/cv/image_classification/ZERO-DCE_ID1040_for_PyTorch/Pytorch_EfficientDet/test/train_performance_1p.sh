#!/bin/bash

cur_path=`pwd`/../
cd $cur_path
rm -rf ./pretraining_output

if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
export MODEL_NAME=000000-bootstrap
start=$(date +%s)
python3 train.py \
  --c=0 \
  --batch_size=8 \
  --optim sgd \ 
  --lr 8e-2 
"${@:1}" >$cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

#step_sec=`grep -a 'INFO:tensorflow:global_step/sec: ' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $2}'`
Performance=`grep -rn "golbal_step/sec" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk 'BEGIN{FS=":";total} {total=total+1000/$5;} END {print "%.2f",total/6'`
#echo "Final Precision MAP : $average_prec"
echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2etime"
