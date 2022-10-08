#!/bin/bash
currentDir=$(cd "$(dirname "$0")";pwd)/..

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
export RANK_SIZE=8

# 数据集路径,保持为空,不需要修改
data_path1=""
data_path2=""

#网络名称,同目录名称,需要模型审视修改
Network="SwinIR"

#训练batch_size,,需要模型审视修改
batch_size=32



#参数校验，不需要修改
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path1* ]];then
        data_path1=`echo ${para#*=}`
    elif [[ $para == --data_path2* ]];then
        data_path2=`echo ${para#*=}`
    fi
done



# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
ASCEND_DEVICE_ID=8
echo "device id is ${ASCEND_DEVICE_ID}"

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

#进入训练脚本目录，需要模型审视修改
cd $cur_path/

#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
    mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
else
    mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
fi

#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${cur_path}/test/env_npu.sh
fi

#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
python3 -m torch.distributed.launch \
       --nproc_per_node=8 \
       --master_port=1158 main_train_psnr.py \
       --opt options/swinir/train_swinir_sr_classical.json  \
       --num 8 \
       --bs ${batch_size} \
       --data_path1 ${data_path1} \
       --data_path2 ${data_path2} \
       --dist True > test/output/$ASCEND_DEVICE_ID/train_full_8p.log   2>& 1

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
echo "end_time: ${end_time}"
e2e_time=$(( $end_time - $start_time ))

#最后一个迭代FPS值
step_time=`grep -a 'time:'  ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_full_8p.log|awk -F "time:" '{print $NF}'|awk 'END {print}'| awk -F " " '{print $1}'`

FPS=`awk 'BEGIN{printf "%.2f\n", 1.0*'${batch_size}'/'${step_time}'}'`
#最后一个迭代loss值
loss=`grep -a 'loss:'  ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_full_8p.log | awk -F "G_loss: " '{print $NF}'| awk 'END {print}' | awk -F " " '{print $1}'`

#打印，不需要修改
echo "ActualFPS : $FPS"
echo "ActualLoss : ${loss}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep -a 'loss:'  ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_full_8p.log | awk -F "G_loss: " '{print $NF}' | awk -F " " '{print $1}' >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

grep -a 'time:'  ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_full_8p.log | awk -F "time:" '{print $NF}' | awk -F " " '{print $1}' >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_FPS.txt

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${FPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${loss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
