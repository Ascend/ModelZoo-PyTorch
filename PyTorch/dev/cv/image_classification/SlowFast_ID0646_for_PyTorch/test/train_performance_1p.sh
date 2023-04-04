#!/bin/bash
#当前路径,不需要修改
cur_path=`pwd`

for para in $*
do
	if [[ $para == --data_path* ]];then
		data_path=`echo ${para#*=}`
    elif [[ $para == --conda_name* ]];then
        conda_name=`echo ${para#*=}`
        i=`pip3 list | grep torch-npu|awk 'END {print $2}'`
        j="1.8"
        result=$(echo $i | grep "${j}")
        if [[ "$result" != "" ]]
        then
            source set_conda.sh
            source activate $conda_name
        else
            source ${cur_path}/set_conda.sh --conda_name=py8_1.11
            source activate py8_1.11
        fi
	fi
done

export ASCEND_SLOG_PRINT_TO_STDOUT=0




#集合通信参数,不需要修改
export RANK_SIZE=1
export JOB_ID=10087
export LIBRARY_PATH=/usr/local/ffmpeg/lib:$LIBRARY_PATH
export CPATH=/usr/local/ffmpeg/include:$CPATH
export NPU_CALCULATE_DEVICE=$ASCEND_DEVICE_ID

RANK_ID_START=0
# 数据集路径,保持为空,不需要修改
data_path=""

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="SlowFast_ID0646_for_PyTorch"
# 训练的batch_size
batch_size=2
#训练epoch
train_epochs=2
#训练step
train_steps=100


#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
autotune=False
PREC=""

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump                  if or not over detection, default is False
    --data_dump_flag         data dump flag, default is False
    --data_dump_step             data dump step, default is 10
    --profiling                  if or not profiling for performance debug, default is False
    --data_path                  source data of training
    -h/--help                    show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --precision_mode* ]];then
        apex_opt_level=`echo ${para#*=}`
                    if [[ $apex_opt_level != "O1" ]] && [[ $apex_opt_level != "O2" ]] && [[ $apex_opt_level != "O3" ]]; then
                            echo "[ERROR] para \"precision_mode\" must be config O1 or O2 or O3"
                            exit 1
                    fi
        PREC="--apex --apex-opt-level "$apex_opt_level

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
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#冒烟看护
cd $cur_path/../tools/
sed -i "s|pass|break|g" train_net.py

#训练开始时间，不需要修改
start_time=$(date +%s)

# 该网络训练脚本需要的文件夹定义 需要修改
cd ..
python3 setup.py install

#进入训练脚本目录，需要模型审视修改
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID
    
    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi

    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    nohup python3 tools/run_net.py --cfg configs/Charades/SLOWFAST_16x8_R50.yaml \
	    TRAIN.BATCH_SIZE ${batch_size} \
	    NUM_GPUS 1 \
            SOLVER.MAX_EPOCH ${train_epochs} \
	    TRAIN.CHECKPOINT_FILE_PATH $data_path/checkpoints/SLOWFAST_8x8_R50.pkl \
	    DATA.PATH_TO_DATA_DIR $data_path/charades \
	    DATA.PATH_PREFIX $data_path/charades/Charades_v1_rgb \
            $PREC > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done 
wait

#参数回改
cd $cur_path/../tools/
sed -i "s|break|pass|g" train_net.py

#退出conda环境
conda deactivate

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "Final Training Duration sec : $e2e_time"

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=`grep "train_iter"  $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "dt" '{print $2}' |awk -F ": " '{print $2}'|awk -F "," '{print $1}'|tail -n+2|awk '{sum+=$1} END {print"",sum/NR}'|sed s/[[:space:]]//g`

FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${TrainingTime}'}'`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"
#输出编译时间
CompileTime=`grep step_time $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log| head -2 |awk -F "step_time = " '{print $2}' | awk '{sum+=$1} END {print"",sum}' |sed s/[[:space:]]//g`

#输出训练精度,需要模型审视修改
train_accuracy="None"
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

###下面字段用于冒烟看护
BatchSize=${batch_size}
#设备类型，自动获取
DeviceType=`uname -m`
#用例名称，自动获取
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "train_iter" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F "loss" '{print $2}' |awk -F ": " '{print $2}'|awk -F "," '{print $1}' >$cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到CaseName.log中，此处无需修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CompileTime = ${CompileTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log