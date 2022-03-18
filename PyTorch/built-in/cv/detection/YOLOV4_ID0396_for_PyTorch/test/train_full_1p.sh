#!/bin/bash
cur_path=`pwd`
export ASCEND_SLOG_PRINT_TO_STDOUT=0
ls /npu/traindata/coco_txl >1.txt
ls /npu/traindata/coco_txt/images >2.txt
ls /npu/traindata/coco_txl/images/train2017 >3.txt
################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="YOLOV4_ID0396_for_PyTorch"
# 训练batch_size
batch_size=32
# 训练使用的npu卡数
export RANK_SIZE=1
# 数据集路径,保持为空,不需要修改
data_path=""
# 训练epoch
train_epochs=300
# 图片大小
image_size=608
# 指定训练所使用的npu device卡id
device_id=0

# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi
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

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi


#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

if [ -d $data_path/../coco_txl/COCO2017/images/train2017/000000000009.jpg ];then
        echo "NO NEED UNTAR"
else
    mkdir -p $data_path/../coco_txl
        tar -zxvf $data_path/COCO2017.tar.gz -C  $data_path/../coco_txl/
rm -rf $data_path/../coco_txl/COCO2017/labels/*.cache
fi
wait

sed -i "s|./coco/train2017.txt|$data_path/../coco_txl/COCO2017/train2017.txt|g" data/coco.yaml
sed -i "s|./coco/val2017.txt|$data_path/../coco_txl/COCO2017/val2017.txt|g" data/coco.yaml
sed -i "s|./coco/testdev2017.txt|$data_path/../coco_txl/COCO2017/testdev2017.txt|g" data/coco.yaml
sed -i "s|./coco/annotations/instances_val|$data_path/../coco_txl/COCO2017/annotations/instances_val|g" test.py

#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)

taskset -c 0-47 python3.7 train.py \
                --device_id ${ASCEND_DEVICE_ID} \
                --img $image_size $image_size \
                --data coco.yaml \
                --cfg cfg/yolov4.cfg \
                --weights '' \
                --name yolov4 \
                --batch-size ${batch_size} \
                --epochs=${train_epochs} \
                --amp \
                --opt-level O1 \
                --loss_scale 128 \
                --notest > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait

# 训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

sed -i "s|$data_path/../coco_txl/COCO2017/train2017.txt|./coco/train2017.txt|g" data/coco.yaml
sed -i "s|$data_path/../coco_txl/COCO2017/val2017.txt|./coco/val2017.txt|g" data/coco.yaml
sed -i "s|$data_path/../coco_txl/COCO2017/testdev2017.txt|./coco/testdev2017.txt|g" data/coco.yaml
sed -i "s|$data_path/../coco_txl/COCO2017/annotations/instances_val|./coco/annotations/instances_val|g" test.py

# 结果打印，不需要修改
echo "------------------ Final result ------------------"
# 输出性能FPS，需要模型审视修改
FPS=`grep -a 'FPS'  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $4}'|awk 'END {print}'`
# 打印，不需要修改
echo "Final Performance images/sec : $FPS"

# 输出训练精度,需要模型审视修改
train_accuracy=`grep -a 'Precision' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'NR==1{print}'|awk '{print $NF}'`
# 打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

# 性能看护结果汇总
# 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

# 从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
cat ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|tr '\r' '\n'|grep "${image_size}:"|awk -F " " '{print $6}' >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

# 最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
