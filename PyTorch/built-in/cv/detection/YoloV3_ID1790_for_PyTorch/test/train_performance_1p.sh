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
Network="YoloV3_ID1790_for_PyTorch"

#训练batch_size,,需要模型审视修改
batch_size=64

# 指定训练所使用的npu device卡id
device_id=0

#适配profiling，默认为False
profiling=False
stop_step=100

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --conda_name* ]];then
      conda_name=`echo ${para#*=}`
      echo "PATH TRAIN BEFORE: $PATH"
      i=`pip3 list | grep torch-npu|awk 'END {print $2}'`
      j="1.8"
      result=$(echo $i | grep "${j}")
      if [[ "$result" != "" ]]
      then
          source ${test_path_dir}/set_conda.sh --conda_name=$conda_name
          source activate $conda_name
      else
          source ${test_path_dir}/set_conda.sh --conda_name=py1_1.11
          source activate py1_1.11
      fi
      echo "PATH TRAIN AFTER: $PATH"
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
    elif [[ $para == --stop_step* ]];then
        stop_step=`echo ${para#*=}`
    fi
done

if [[ $profiling == "GE" ]];then
    export GE_PROFILING_TO_STD_OUT=1
    profiling=True
elif [[ $profiling == "CANN" ]];then
    profiling=True
fi

#校验是否传入data_path,不需要修改
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

#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source  ${test_path_dir}/env_npu.sh
else
    pip3 install -v -e .
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

#训练开始时间，不需要修改
start_time=$(date +%s)

sed -i "s|total_epochs = 273|total_epochs = 1|g" configs/yolo/yolov3_d53_mstrain-608_273e_coco.py
sed -i "s|data/coco/|$data_path/|g" configs/yolo/yolov3_d53_mstrain-608_273e_coco.py

#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
KERNEL_NUM=$(($(nproc)/8))
PID_START=$((KERNEL_NUM * ASCEND_DEVICE_ID))
PID_END=$((PID_START + KERNEL_NUM - 1))
taskset -c $PID_START-$PID_END python3.7 ./tools/train.py configs/yolo/yolov3_d53_320_273e_coco.py \
    --cfg-options optimizer.lr=0.001 data.samples_per_gpu=${batch_size} \
    --seed 0  \
    --local_rank 0 \
    --precision_mode ${precision_mode} \
    --profiling ${profiling} \
    --stop_step ${stop_step} \
    --npu_ids ${device_id} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait
sed -i "s|$data_path/|data/coco/|g" configs/yolo/yolov3_d53_mstrain-608_273e_coco.py
sed -i "s|total_epochs = 1|total_epochs = 273|g" configs/yolo/yolov3_d53_mstrain-608_273e_coco.py
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
time=`grep -a 'Epoch'  $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "time: " '{print $2}'|awk -F "," '{print $1}'|awk 'END {print}'|sed 's/.$//'`
FPS=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'/'${time}'}'`

#输出CompileTime
time2=`grep -a 'Epoch'  $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|grep time|head -n 1|awk -F "time: " '{print $2}'|awk -F "," '{print $1}'|sed '/^$/d'|sed 's/.$//'`
CompileTime=`awk 'BEGIN{printf "%.2f\n", ('${time2}'-'${time}')*50}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
if [[ $precision_mode == "must_keep_origin_dtype" ]];then
        CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'fp32'_'perf'
else
        CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'
fi

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${time}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep -a Epoch $test_path_dir/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep eta:|awk -F "loss: " '{print $NF}' | awk -F " " '{print $1}'|awk 'END {print}'|sed 's/.$//' >> $test_path_dir/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
