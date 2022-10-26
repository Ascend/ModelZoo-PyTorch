#!/bin/bash

ls /npu/traindata/VOC2012 >1.txt
ls /npu/traindata/VOC2012/ImageSets >2.txt
ls /npu/traindata/VOCdevkit/VOC2012/ImageSets/Main >3.txt
#当前路径,不需要修改
cur_path=`pwd`
export ASCEND_GLOBAL_LOG_LEVEL=1
export NPU_CALCULATE_DEVICE=$ASCEND_DEVICE_ID
#集合通信参数,不需要修改
export RANK_SIZE=1
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#网络名称,同目录名称,需要模型审视修改
Network="SSD-MobileNet_ID1936_for_PyTorch"

#训练batch_size,,需要模型审视修改
batch_size=8

#二进制开关
bin_mode=False
bin_analysis=False

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --bin_mode* ]];then
        bin_mode="True"
    elif [[ $para == --bin_analysis* ]];then
        bin_analysis="True"
    fi
done

PREC=""
if [[ $precision_mode == "amp" ]];then
    PREC="--apex"
fi
#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#修改模糊编译写法
if [ $bin_mode == "True" ];then
    sed -i "61itorch.npu.set_compile_mode(jit_compile=False)" ${cur_path}/train.py
    line=`grep "torch.npu.set_compile_mode(jit_compile=False)" ${cur_path}/train.py -n | awk -F ':' '{print $1}'`
    line=$[ $line+1 ]
    sed -i "${line}itorch.npu.set_option(option)" ${cur_path}/train.py
    sed -i "${line}ioption['ACL_OP_COMPILER_CACHE_MODE'] = 'disable'" ${cur_path}/../train.py
    sed -i "${line}ioption = {}" ${cur_path}/train.py
fi

if [ -d $data_path/VOCdevkit ];then
        echo "NO NEED TARZXVF"
else
        tar -zxvf $data_path/VOCdevkit.tar.gz -C $data_path/
fi
wait



#训练开始时间，不需要修改

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../
#冒烟
sed -i "s|: 20|: 1|g" ${cur_path}/../config.json
#修改数据集路径
sed -i "s|"1111"|"$data_path/VOCdevkit/VOC2007"|g" ${cur_path}/../config.json
sed -i "s|"2222"|"$data_path/VOCdevkit/VOC2012"|g" ${cur_path}/../config.json

#mkdir -p /root/.cache/torch/checkpoints
#cp $data_path/*.pth  /root/.cache/torch/checkpoints/
start_time=$(date +%s)
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID

    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi

    # 绑核，不需要的绑核的模型删除，需要的模型审视修改
    #let a=RANK_ID*12
    #let b=RANK_ID+1
    #let c=b*12-1

    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改

    nohup python3 train.py config.json $PREC > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done
wait

#conda deactivate
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

sed -i "s|"$data_path/VOCdevkit/VOC2007"|"1111"|g" ${cur_path}/../config.json
sed -i "s|"$data_path/VOCdevkit/VOC2012"|"2222"|g" ${cur_path}/../config.json

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep "FPS = " $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "FPS = " '{print $2}'|awk -F ", step_time" '{print $1}' |tail -n+2 |awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
#train_accuracy=`grep "acc = " $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|grep -v mlp_log|awk 'END {print $5}'| sed 's/,//g' |cut -c 1-5`
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"
#echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'
#修改二进制用例名称
if [ $bin_mode == "True" ];then
    CaseName=$CaseName"_binary"
fi

#获取二进制支持算子
if [ $bin_analysis == "True" ];then
    cmd1=`ls -l /usr/local/Ascend/CANN-1.82/opp/op_impl/built-in/ai_core/tbe/kernel/config/ascend910|grep -v total|awk -F " " '{print $9}'|awk -F "." '{print $1}'`
    echo "cmd1=$cmd1" >> ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log
fi

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`
#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "Batch Time" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "Loss " '{print $2}'|awk -F " " '{print$1}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
for i in $(seq 1 4); do sed -i '$d' $cur_path/output/$ASCEND_DEVICE_ID/train_*.log ;done;
