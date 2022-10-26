#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size resume RANK_SIZE
# 网络名称，同目录名称
Network="StyleGAN2-ADA-Pytorch"
# 使用的npu卡数
npu_num=8

# 参数校验，pth_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
args=$(getopt -o h --long help,data_path:,pth_path: -n "$0" -- "$@")
eval set -- "$args"
while true ; do
    case "$1" in 
    -h|--help)
        echo "Options:"
        echo "    --data_path specify the path of dataset"
        echo "    --pth_path specif the path of model"
        h="h"
        shift
        ;;
    --data_path)
        d=$2
        shift 2
        ;;
    --pth_path)
        n=$2
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "Error!"
        exit 1
        ;;
    esac
done

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" = x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi

#################创建日志输出目录，不需要修改#################
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

#################启动FID计算脚本#################
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi
if [ -n "$n" ]; then
    #开始时间，不需要修改
    start_time=$(date +%s)
    if [ -n "$d" ] ; then
        python calc_metrics.py --metrics=fid50k_full --gpus=$npu_num --data=$d --network=$n > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
        else
        python calc_metrics.py --metrics=fid50k_full --gpus=$npu_num --network=$n > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    fi
    
    wait
    
    ##################获取训练数据################
    # 结束时间，不需要修改
    end_time=$(date +%s)
    e2e_time=$(( $end_time - $start_time ))
    
    #结果打印，不需要修改
    echo "------------------ Final result ------------------"
    fid=$(grep "result" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk '{print $3}' | awk -F '}' '{print $1}')
    echo "FID calculation result: $fid"
    echo "E2E Training Duration sec : $e2e_time"
    
    # 训练用例信息，不需要修改
    DeviceType=`uname -m`
    CaseName=${Network}_${npu_num}'p'_'fid'
    
    #关键信息打印到${CaseName}.log中，不需要修改
    echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "RankSize = ${npu_num}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "FID = ${fid}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
    echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
    
    else
        if [ -z "$h" ] ; then
            echo "Error: --pth_path: Path must point to a directory or zip"
            echo "Try '--help' for help"
        fi
fi
