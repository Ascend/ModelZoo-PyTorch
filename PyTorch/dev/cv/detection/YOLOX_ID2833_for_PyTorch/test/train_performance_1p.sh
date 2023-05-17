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
export RANK_SIZE=1

# 数据集路径,保持为空,不需要修改
data_path=""

#网络名称,同目录名称,需要模型审视修改
Network="YOLOX_ID2833_for_PyTorch"

#训练batch_size,,需要模型审视修改
batch_size=8
#总训练Epoch数
total_epoch=1
#val间隔数，多少个Epoch
val_epoch=1

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
    if [[ $para == --conda_name* ]];then
      conda_name=`echo ${para#*=}`
      source set_conda1
      source activate $conda_name
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
device_id=$ASCEND_DEVICE_ID
#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source  ${test_path_dir}/env_npu.sh
fi


#进入训练脚本目录，需要模型审视修改
cd $cur_path

#创建DeviceID输出目录，不需要修改
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/ckpt
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/ckpt
fi
chmod +x ${cur_path}/tools/dist_train.sh


#runtime 2.0 enable
export ENABLE_RUNTIME_V2=1
echo "Runtime 2.0 $ENABLE_RUNTIME_V2"

sed -i "s|max_epochs = [0-9]\{1,3\}|max_epochs = $total_epoch|g" configs/yolox/yolox_s_8x8_300e_coco.py
sed -i "s|data/coco/|$data_path/|g" configs/yolox/yolox_s_8x8_300e_coco.py
sed -i "s|interval = [0-9]\{1,3\}|interval = $val_epoch|g" configs/yolox/yolox_s_8x8_300e_coco.py
sed -i "s|annotations/instances_train2017.json|annotations/MINIinstances_train2017.json|g" configs/yolox/yolox_s_8x8_300e_coco.py

#训练开始时间，不需要修改
start_time=$(date +%s)
#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
PORT=29500 ./tools/dist_train.sh configs/yolox/yolox_m_8x8_300e_coco.py 1  \
    --cfg-options data.persistent_workers=True log_config.interval=50  \
    --no-validate  \
    --launcher none  \
    --local_rank=${device_id}  \
    --gpu-id=${device_id} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait
sed -i "s|max_epochs = $total_epoch|max_epochs = 300|g" configs/yolox/yolox_s_8x8_300e_coco.py
sed -i "s|$data_path/|data/coco/|g" configs/yolox/yolox_s_8x8_300e_coco.py
sed -i "s|interval = $val_epoch|interval = 10|g" configs/yolox/yolox_s_8x8_300e_coco.py
sed -i "s|annotations/MINIinstances_train2017.json|annotations/instances_train2017.json|g" configs/yolox/yolox_s_8x8_300e_coco.py

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
time=`grep -a ', time'  $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "time: " '{print $2}'|awk -F "," '{print $1}'|tail -n 10|awk '{sum+=$1} END {print sum/NR}'`
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'/'${time}'}'`
compile_time=`grep -a ', time'  $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "time: " '{print $2}'|awk -F "," '{print $1}'|head -n 1|awk '{sum+=$1} END {print sum}'`
CompileTime=`awk 'BEGIN{print ('$compile_time'-'$time')*50}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${time}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep -a Epoch $test_path_dir/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep eta:|tr -d '\b\r'|grep -Eo "loss: [0-9]*\.[0-9]*"|awk -F " " '{print $2}' >> $test_path_dir/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
echo "ActualLoss = ${ActualLoss}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CompileTime = ${CompileTime}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
#退出anaconda环境
if [ -n "$conda_name" ];then
    echo "conda $conda_name deactivate"
    conda deactivate
fi
