#!/bin/bash

#当前路径，不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
export WORLD_SIZE=1

#网络名称,同目录名称,需要模型审视修改
Network="AdvancedEAST"

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_diename=${cur_path##*/}
if [ x"${cur_path_last_diename}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi
echo ${pwd}

#################创建日志输出目录，不需要修改#################
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

start_time=$(date +%s)
source test/env.sh

if [ -n "$*" ]
then
    SIZES=$*
else
    SIZES="256 384 512 640 736"
fi

for SIZE in $SIZES
do
    nohup python3 -u train.py --size $SIZE --apex --epoch_num 3 --val_interval 1 > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${SIZE}.log 2>&1
    sleep 5s
done

wait 

##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
FPS_256=`grep -a 'FPS:' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_256.log|awk -F " " '{print $4}'|awk 'END {print}'` 
FPS_384=`grep -a 'FPS:' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_384.log|awk -F " " '{print $4}'|awk 'END {print}'` 
FPS_512=`grep -a 'FPS:' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_512.log|awk -F " " '{print $4}'|awk 'END {print}'` 
FPS_640=`grep -a 'FPS:' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_640.log|awk -F " " '{print $4}'|awk 'END {print}'` 
FPS_736=`grep -a 'FPS:' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_736.log|awk -F " " '{print $4}'|awk 'END {print}'` 

echo "Final Performance FPS : 256:${FPS_256}, 384:${FPS_384}, 512:${FPS_512}, 640:${FPS_640}, 736:${FPS_736}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
DeviceType=`uname -m`
CaseName=${Network}_${WORLD_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
AvgFPS=${FPS_736}

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${WORLD_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "AvgFPS = ${AvgFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
