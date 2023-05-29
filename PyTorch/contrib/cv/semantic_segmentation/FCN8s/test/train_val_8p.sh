#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
export RANK_SIZE=8

# 数据集路径,保持为空,不需要修改
data_path=""

#网络名称,同目录名称,需要模型审视修改
Network="FCN_val"

#训练batch_size,,需要模型审视修改
batch_size=16

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --device_id* ]];then
        device_id=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
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
    export ASCEND_DEVICE_ID=0
    echo "device id is ${ASCEND_DEVICE_ID}"
fi

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

#进入训练脚本目录，需要模型审视修改
cd $cur_path/

#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/output/${Network}/${ASCEND_DEVICE_ID} ];then
    rm -rf ${cur_path}/output/${Network}/${ASCEND_DEVICE_ID}
    mkdir -p ${cur_path}/output/${Network}/$ASCEND_DEVICE_ID/
else
    mkdir -p ${cur_path}/output/${Network}/$ASCEND_DEVICE_ID/
fi

#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${cur_path}/env_npu.sh
fi

#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
python3 tools/test.py configs/fcn/fcn_r50-d8_512x512_20k_voc12aug.py \
    ${cur_path}/output/FCN/0,1,2,3,4,5,6,7/ckpt/latest.pth \
    --eval mIoU \
    --options data_root=${data_path} data.train.data_root=${data_path} data.val.data_root=${data_path} data.test.data_root=${data_path} \
    > ${cur_path}/output/${Network}/${ASCEND_DEVICE_ID}/val_${ASCEND_DEVICE_ID}.log 2>&1 &

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
echo "end_time: ${end_time}"
e2e_time=$(( $end_time - $start_time ))

total_images=1449
FPS=`awk 'BEGIN{printf "%.2f\n", '${total_images}'/'${e2e_time}'}'`
echo "FPS : $FPS"

mIoU=`grep -a 'global |' ${cur_path}/output/${Network}/${ASCEND_DEVICE_ID}/val_${ASCEND_DEVICE_ID}.log|awk -F "global |" '{print $NF}'|awk -F "| " '{print $2}'`
echo "mIoU : $mIoU"

echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
ValTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/${Network}/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/${Network}/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/${Network}/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/${Network}/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/${Network}/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/${Network}/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ValTime = ${ValTime}" >> $cur_path/output/${Network}/$ASCEND_DEVICE_ID/${CaseName}.log
echo "mIoU = ${mIoU}" >> $cur_path/output/${Network}/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2EValTime = ${e2e_time}" >> $cur_path/output/${Network}/$ASCEND_DEVICE_ID/${CaseName}.log




