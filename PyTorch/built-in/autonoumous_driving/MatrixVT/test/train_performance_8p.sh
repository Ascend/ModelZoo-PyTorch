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
RANK_ID_START=0


# 数据集路径,保持为空,不需要修改
data_path=""

#删除outputs文件夹，不需要修改
if [ -d ${cur_path}/outputs/ ];then
    rm -rf ${cur_path}/outputs/
ckpt_path="./outputs/matrixvt_bev_depth_lss_r50_256x704_128x128_24e_ema_cbgs/lightning_logs/version_0/23.pth"
fi
#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="MatrixVT_for_PyTorch"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=8
#学习率
learning_rate=0.000003125


#设置环境变量，不需要修改
ASCEND_DEVICE_ID=0


#创建DeviceID输出目录，不需要修改
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/ckpt

else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/ckpt
fi

#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source  ${test_path_dir}/env_npu.sh
fi

#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/

nohup python3 ./bevdepth/exps/nuscenes/MatrixVT/matrixvt_bev_depth_lss_r50_256x704_128x128_24e_ema.py \
          --seed 0 \
          --learning_rate ${learning_rate} \
          --max_epoch ${train_epochs} \
          --amp_backend 'native' \
          --gpus ${RANK_SIZE} \
          --precision 16 \
          --batch_size_per_device ${batch_size}  > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))


#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
step_time=`grep -o 'step_time=\s*[0-90]\+\(\.[0-9]\+\)\?\s*' $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log| sed 's/step_time= //g' | tail -n 50 | awk '{sum += $1} END {avg = sum / NR; print avg}'`
FPS=`awk 'BEGIN{printf "%d\n", '$batch_size'/'$step_time'*'$RANK_SIZE'}'`
#排除功能问题导致计算溢出的异常，增加健壮性
if [ x"${FPS}" == x"2147483647" ] || [ x"${FPS}" == x"-2147483647" ];then
    FPS=""
fi
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*'${RANK_SIZE}'*1000/'${FPS}'}'`



#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $test_path_dir/output/$ASCEND_DEVICE_ID/${CaseName}.log
