#!/bin/bash

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

#集合通信参数,不需要修改
export RANK_SIZE=8

# 数据集路径,保持为空,不需要修改
data_path=""


#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="Faster_Mask_RCNN_ID0101_for_PyTorch"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=64
#训练step
train_steps=3000

# for multi node setting
nnodes=1
node_rank=0
local_addr=127.0.0.1
master_addr=127.0.0.1
master_port=23333

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --over_dump* ]];then
        over_dump=`echo ${para#*=}`
        over_dump_path=${cur_path}/test/output/overflow_dump
        mkdir -p ${over_dump_path}
    elif [[ $para == --data_dump_flag* ]];then
        data_dump_flag=`echo ${para#*=}`
        data_dump_path=${cur_path}/test/output/data_dump
        mkdir -p ${data_dump_path}
    elif [[ $para == --data_dump_step* ]];then
        data_dump_step=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
        profiling_dump_path=${cur_path}/test/output/profiling
        mkdir -p ${profiling_dump_path}
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
	elif [[ $para == --batch_size* ]]; then
		batch_size=$(echo ${para#*=})
	elif [[ $para == --nnodes* ]]; then
		nnodes=$(echo ${para#*=})
	elif [[ $para == --node_rank* ]]; then
		node_rank=$(echo ${para#*=})
	elif [[ $para == --local_addr* ]]; then
		local_addr=$(echo ${para#*=})
	elif [[ $para == --master_addr* ]]; then
		master_addr=$(echo ${para#*=})
	elif [[ $para == --master_port* ]]; then
		master_port=$(echo ${para#*=})
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

# 指定训练所使用的npu device卡id
device_id=0

# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
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
        mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/ckpt
fi


#修改参数
sed -i "s|\"coco_2017_train\": (\"coco/train2017\", \"coco/annotations/instances_train2017.json\")|\"coco_2017_train\": (\"$data_path/coco/train2017\", \"$data_path/coco/annotations/instances_train2017.json\")|g" $cur_path/detectron2/data/datasets/builtin.py
sed -i "s|\"coco_2017_val\": (\"coco/val2017\", \"coco/annotations/instances_val2017.json\")|\"coco_2017_val\": (\"$data_path/coco/val2017\", \"$data_path/coco/annotations/instances_val2017.json\")|g" $cur_path/detectron2/data/datasets/builtin.py
sed -i "s|WEIGHTS: \"detectron2://ImageNetPretrained/MSRA/R-101.pkl\"|WEIGHTS: \"$data_path/R-101.pkl\"|g" $cur_path/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml
wait

cd $cur_path/
#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source  ${test_path_dir}/env_npu.sh
fi
cd $cur_path

python3.7 setup.py build develop > $cur_path/log.txt

#训练开始时间，不需要修改
start_time=$(date +%s)

# 多机多卡
export HCCL_IF_IP=$local_addr

nohup python3.7 tools/train_net.py \
        --config-file  configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml \
        --master-addr $master_addr \
        --master-port $master_port \
        --device-ids 0 1 2 3 4 5 6 7 \
        --num-gpus 8 \
        --num-machines $nnodes \
        --machine-rank $node_rank \
        AMP 1 \
        OPT_LEVEL O2 \
        LOSS_SCALE_VALUE 64 \
        SOLVER.IMS_PER_BATCH $batch_size \
        SOLVER.MAX_ITER $train_steps \
        SEED 1234 \
        MODEL.RPN.NMS_THRESH 0.8 \
        MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO 2 \
        MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO 2 \
        DATALOADER.NUM_WORKERS 8 \
        SOLVER.BASE_LR 0.02 > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait
#修改参数
sed -i "s|\"coco_2017_train\": (\"$data_path/coco/train2017\", \"$data_path/coco/annotations/instances_train2017.json\")|\"coco_2017_train\": (\"coco/train2017\", \"coco/annotations/instances_train2017.json\")|g" $cur_path/detectron2/data/datasets/builtin.py
sed -i "s|\"coco_2017_val\": (\"$data_path/coco/val2017\", \"$data_path/coco/annotations/instances_val2017.json\")|\"coco_2017_val\": (\"coco/val2017\", \"coco/annotations/instances_val2017.json\")|g" $cur_path/detectron2/data/datasets/builtin.py
sed -i "s|WEIGHTS: \"$data_path/R-101.pkl\"|WEIGHTS: \"detectron2://ImageNetPretrained/MSRA/R-101.pkl\"|g" $cur_path/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml
wait
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep FPS $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $NF}'|awk '{sum+=$1} END {print  sum/NR}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep "Average Precision" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "=" '{print $NF}'|head -1`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep total_loss $test_path_dir/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F 'total_loss: ' '{print $2}'|awk '{print $1}' > $test_path_dir/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $test_path_dir/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
