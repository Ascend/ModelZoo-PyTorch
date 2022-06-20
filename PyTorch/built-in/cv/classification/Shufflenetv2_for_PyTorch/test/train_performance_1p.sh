#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
#保证rank table file 文件rank_table_8p.json存放在和test同级的configs目录下
export RANK_SIZE=1

#RANK_TABLE_FILE=${cur_path}/../configs/rank_table_8p.json
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数 需要模型审视修改
#网络名称，同目录名称
Network="Shufflenetv2_ID0099_for_PyTorch"
#训练epoch
train_epochs=2
#训练batch_size
batch_size=1536

#TF2.X独有，不需要修改
#export NPU_LOOP_SIZE=${train_steps}

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
autotune=False
#二进制开关
bin_mode=False
bin_analysis=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_8p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode           precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		           if or not over detection, default is False
    --data_dump_flag		   data dump flag, default is 0
    --data_dump_step		   data dump step, default is 10
    --profiling		           if or not profiling for performance debug, default is False
    --autotune                 whether to enable autotune, default is False
    --data_path		           source data of training
    -h/--help		           show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --over_dump* ]];then
        over_dump=`echo ${para#*=}`
        over_dump_path=${cur_path}/output/overflow_dump
        mkdir -p ${over_dump_path}
    elif [[ $para == --data_dump_flag* ]];then
        data_dump_flag=`echo ${para#*=}`
        data_dump_path=${cur_path}/output/data_dump
        mkdir -p ${data_dump_path}
    elif [[ $para == --data_dump_step* ]];then
        data_dump_step=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
        profiling_dump_path=${cur_path}/output/profiling
        mkdir -p ${profiling_dump_path}
    elif [[ $para == --autotune* ]];then
        autotune=`echo ${para#*=}`
		export autotune=$autotune
        mv $install_path/fwkacllib/data/rl/Ascend910/custom $install_path/fwkacllib/data/rl/Ascend910/custom_bak
        mv $install_path/fwkacllib/data/tiling/Ascend910/custom $install_path/fwkacllib/data/tiling/Ascend910/custom_bak
        autotune_dump_path=${cur_path}/output/autotune_dump
        mkdir -p ${autotune_dump_path}/GA
        mkdir -p ${autotune_dump_path}/rl
        cp -rf $install_path/fwkacllib/data/tiling/Ascend910/custom ${autotune_dump_path}/GA/
        cp -rf $install_path/fwkacllib/data/rl/Ascend910/custom ${autotune_dump_path}/RL/
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --bin_mode* ]];then
        bin_mode="True"
    elif [[ $para == --bin_analysis* ]];then
        bin_analysis="True"
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#修改模糊编译写法
if [ $bin_mode == "True" ];then
    sed -i "46itorch.npu.set_compile_mode(jit_compile=False)" ${cur_path}/../8p_main_med.py
    line=`grep "torch.npu.set_compile_mode(jit_compile=False)" ${cur_path}/../8p_main_med.py -n | awk -F ':' '{print $1}'`
    line=$[ $line+1 ]
    sed -i "${line}itorch.npu.set_option(option)" ${cur_path}/../8p_main_med.py
    sed -i "${line}ioption['NPU_FUZZY_COMPILE_BLACKLIST'] = 'Slice'" ${cur_path}/../8p_main_med.py
    sed -i "${line}ioption = {}" ${cur_path}/../8p_main_med.py
fi

#设置二进制变量
if [ $bin_analysis == "True" ];then
    #增加编译缓存设置
    line=`grep "torch.npu.set_option(option)" ${cur_path}/../8p_main_med.py -n | awk -F ':' '{print $1}'`
    sed -i "${line}ioption['ACL_OP_COMPILER_CACHE_MODE'] = 'disable'" ${cur_path}/../8p_main_med.py
fi

#训练开始时间，不需要修改 
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID
    export DEVICE_ID=$RANK_ID
    
    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
        mkdir -p ${cur_path}/output/overflow_dump
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
        mkdir -p ${cur_path}/output/overflow_dump
    fi
    over_dump_path=${cur_path}/output/overflow_dump
    
    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
  
    python3 8p_main_med.py \
        --data=$data_path \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49  \
        --workers=128 \
        --learning-rate=0.75 \
        --print-freq=10 \
        --eval-freq=5 \
        --arch=shufflenet_v2_x1_0  \
        --dist-url='tcp://127.0.0.1:50000' \
        --dist-backend='hccl' \
        --multiprocessing-distributed \
        --world-size=1 \
        --batch-size=1536 \
        --epochs=2 \
        --warm_up_epochs=1 \
        --rank=0 \
        --amp \
        --momentum=0 \
        --wd=3.0517578125e-05 \
        --device-list=0 \
        --benchmark 0 \
		> ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
        
done 
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))
echo "E2E Training Duration sec : $e2e_time"

#cp -r ${cur_path}/train.log ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log

#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'
if [ $bin_mode == "True" ];then
    CaseName=$CaseName"_binary"
fi

#二进制支持算子
if [ $bin_analysis == "True" ];then
    cmd1=`ls -l /usr/local/Ascend/CANN-1.82/opp/op_impl/built-in/ai_core/tbe/kernel/config/ascend910|grep -v total|awk -F " " '{print $9}'|awk -F "." '{print $1}'`
    echo "cmd1=$cmd1" >> ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log
fi

##获取性能数据
FPS=`grep "FPS@all" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $7}'|tr -d ,| sed s/[[:space:]]//g`
ActualFPS=${FPS}

temp1=`echo "1 * ${batch_size}"|bc`
TrainingTime=`echo "scale=2;${temp1} / ${ActualFPS}"|bc`

ActualLoss=`grep "Loss" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $12}'`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
