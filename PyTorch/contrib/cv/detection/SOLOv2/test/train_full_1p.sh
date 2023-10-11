#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数
export RANK_SIZE=1

# 数据集路径,保持为空,不需要修改
data_path=""
apex="O1"
#网络名称,同目录名称,需要模型审视修改
Network="SOLOv2"

#训练batch_size,,需要模型审视修改
batch_size=2
device_id=0

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --apex* ]];then
        apex=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    fi
done

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

#训练开始时间，不需要修改
start_time=$(date +%s)

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


#创建DeviceID输出目录，不需要修改
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
export NPUID=0
export RANK=0
python3 tools/train.py configs/solov2/solov2_r50_fpn_8gpu_1x.py --opt-level $apex --autoscale-lr --seed 0 --total_epochs 12 \
      --batch_size=${batch_size} --data_root=$data_path --gpu-ids ${device_id} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait
python3 tools/test_ins.py configs/solov2/solov2_r50_fpn_8gpu_1x.py  test/work_dirs/solov2_release_r50_fpn_8gpu_1x/latest.pth --show \
      --gpu-ids ${device_id} --out  results_solo.pkl --eval segm --data_root=$data_path >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a '^FPS:'  $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "FPS: " '{print $NF}'|awk 'NR==1{max=$1;next}{max=max>$1?max:$1}END{print max}'`
#FPS值小数点后保留2位
FPS=${FPS:0:$[`expr index $FPS '.'`+2]}
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a 'maxDets' $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F " " '{print $13}'|head -n 1`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
if [[ $apex == "O0" ]];then
        CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'fp32'_'acc'
else
        CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'
fi

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep Epoch: $test_path_dir/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep eta:|awk -F "loss: " '{print $NF}' | awk -F "," '{print $1}' >> $test_path_dir/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
