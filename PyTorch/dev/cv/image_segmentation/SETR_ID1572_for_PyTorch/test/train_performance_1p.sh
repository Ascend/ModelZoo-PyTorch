#!/bin/bash
cur_path=`pwd`/../
#失败用例打屏
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export ASCEND_DEVICE_ID=6
export COMBINED_ENABLE=1
export TRI_COMBINED_ENABLE=1
export PTCOPY_ENABLE=1
export TASK_QUEUE_ENABLE=1
#参数配置
data_path="./segmentation_car"
precision_mode=amp
epoches=1

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh --data_path ./segmentation_car"
   exit 1
fi

for para in $*
do
   if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
   elif [[ $para == --precision_mode* ]];then
    precision_mode=`echo ${para#*=}`
   fi
done

PREC=""
if [[ $precision_mode == "amp" ]];then
    PREC="--apex"
fi

if [[ $data_path  == "" ]];then
   echo "[Error] para \"data_path \" must be config"
   exit 1
fi
##############执行训练##########
cd $cur_path
if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

start=$(date +%s)
nohup python3 tast_car_seg.py --epoches=$epoches --data_path $data_path \
        --device_id $ASCEND_DEVICE_ID \
        $PREC > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2etime"
sed -i "s|\r|\n|g"  $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 

#定义基本信息
Network="SETR_ID1572_for_PyTorch"
RANK_SIZE=1
BatchSize=3
batch_size=3
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

#获取性能
ActualFPS=`grep "/1357" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log  |awk -F "," '{print$2}'|grep -v "s/it"|tail -n +5|awk -F "it" '{print$1}'|awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g|sed '/^$/d'`
TrainingTime=`awk 'BEGIN{printf "%2f\n",'${BatchSize}'*1000/'${ActualFPS}'}'`
grep "report_loss" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $3}' >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}_loss.log
ActualLoss=`awk 'END {print $1}' $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}_loss.log`
TrainAccuracy=`grep "dice" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|tail -n 1|cut -d "," -f 1|cut -d "(" -f 2`
#输出编译时间
CompileTime=`grep "step_time = " $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | head -2 |awk -F 'step_time = ' '{print $2}'| awk '{sum+=$1} END {print"",sum}' |sed s/[[:space:]]//g`

#关键信息打印到CaseName.log中，此处无需修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${TrainAccuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CompileTime = ${CompileTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log