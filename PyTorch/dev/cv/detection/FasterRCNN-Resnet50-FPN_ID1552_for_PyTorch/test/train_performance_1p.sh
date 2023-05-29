#!/bin/bash
cp -r ../detectron2/engine/train_loop.py /home/tmp/Faster_Mask_RCNN_for_PyTorch/detectron2/engine/train_loop.py
cp -r ../detectron2/engine/defaults.py /home/tmp/Faster_Mask_RCNN_for_PyTorch/detectron2/engine/defaults.py
cp -r ../detectron2/utils/events.py /home/tmp/Faster_Mask_RCNN_for_PyTorch/detectron2/utils/events.py
cp -r ../detectron2/data/dataset_mapper.py /home/tmp/Faster_Mask_RCNN_for_PyTorch/detectron2/data/dataset_mapper.py

#当前路径,不需要修改
cur_path=`pwd`

unset PYTHONPATH
source /usr/local/Ascend/latest/bin/setenv.bash

#集合通信参数,不需要修改
export RANK_SIZE=1
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#训练参数,需要模型审视修改
Network="FasterRCNN-Resnet50-FPN_ID1552_for_PyTorch"
num_train_steps=1000
LR_step_1=480000
LR_step_2=640000
base_lr=0.0025
batch_size=2
ckpt_path=./output

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#配置数据集路径
export DETECTRON2_DATASETS=$data_path

#安装detectron2
cd $cur_path/../
python3 -m pip install -e ./

#训练开始时间，不需要修改
start_time=$(date +%s)

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID

    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/
    fi
done
#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
python3 $cur_path/../tools/train_net.py \
	--config-file $cur_path/../configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml \
	AMP 1 \
	OPT_LEVEL O2 \
	LOSS_SCALE_VALUE 128 \
	MODEL.DEVICE npu:$ASCEND_DEVICE_ID \
	MODEL.WEIGHTS "$data_path/R-50.pkl" \
	MODEL.RPN.NMS_THRESH 0.7 \
	MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO 2 \
	MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO 2 \
	DATALOADER.NUM_WORKERS 8 \
	SOLVER.IMS_PER_BATCH $batch_size \
	SOLVER.BASE_LR $base_lr \
	SOLVER.MAX_ITER $num_train_steps \
	SOLVER.STEPS $LR_step_1,$LR_step_2 > $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a 'Overall'  $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F '(' '{print $2}'|awk '{print $1}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${FPS}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`cat $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep copypaste|tail -1|awk -F ' ' '{print $6}'|awk -F ',' '{print $1}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
DeviceType=`uname -m`
CaseName=${Network}_bs${batch_size}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep total_loss $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F 'total_loss: ' '{print $2}'|awk '{print $1}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${batch_size}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log