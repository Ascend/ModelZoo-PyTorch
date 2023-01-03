#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`/../

#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="ArcFace_ID0852_for_PyTorch"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=32
#训练step
train_steps=null
#学习率
learning_rate=0.1

#TF2.X独有，需要模型审视修改
#export NPU_LOOP_SIZE=${train_steps}

#维测参数，precision_mode需要模型审视修改
#precision_mode="allow_mix_precision"
PREC="--apex --apex-opt-level O2"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1P.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		           if or not over detection, default is False
    --data_dump_flag		     data dump flag, default is False
    --data_dump_step		     data dump step, default is 10
    --profiling		           if or not profiling for performance debug, default is False
    --data_path		           source data of training
    -h/--help		             show help message
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
        over_dump_path=${cur_path}/test/output/overflow_dump
        mkdir -p ${over_dump_path}
    elif [[ $para == --data_dump_flag* ]];then
        data_dump_flag=`echo ${para#*=}`
        data_dump_path=${cur_path}/test/output/data_dump
        mkdir -p ${data_dump_path}
    elif [[ $para == --data_dump_step* ]];then
        data_dump_step=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
        profiling_dump_path=${cur_path}/test/output/profiling
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

#备份自定义知识库
if [ -d "/usr/local/Ascend/opp/data/rl/Ascend910/custom" ]; then
    echo "Backup old rl autotune lib."
    mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/autotune/backup
    mv /usr/local/Ascend/opp/data/rl/Ascend910/custom ${cur_path}/test/output/$ASCEND_DEVICE_ID/autotune/backup
else
    echo "No RL autotune lib exist."   
fi

if [ -d "/usr/local/Ascend/opp/data/tiling/ascend910/custom" ]; then
    echo "Backup old tiling autotune lib."
    mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/autotune_bak/tiling
    mv /usr/local/Ascend/opp/data/tiling/ascend910/custom ${cur_path}/test/output/$ASCEND_DEVICE_ID/autotune/backup
else
    echo "No tiling autotune lib exist."
fi


# dump离线调优数据
# mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/autotune/tune_dump
# chmod +777 ${cur_path}/test/output/$ASCEND_DEVICE_ID/autotune

# export ENABLE_TUNE_DUMP=True
# export TUNE_DUMP_PATH=${cur_path}/test/output/$ASCEND_DEVICE_ID/autotune/tune_dump

#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path
mkdir -p ${cur_path}/result
touch ${cur_path}/result/cur_agedb30_result.mat ${cur_path}/result/cur_cfpfp_result.mat ${cur_path}/result/cur_lfw_result.mat

# 修改性能训练steps，训练时长小于15分钟
sed -i "s|if total_iters == 700: pass|if total_iters == 700: break|g" train.py 
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID

    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/ckpt
    fi
    # 绑核，不需要的绑核的模型删除，需要的模型审视修改
    #let a=RANK_ID*12
    #let b=RANK_ID+1
    #let c=b*12-1

    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path
    nohup python3 train.py \
        --train_root $data_path/webface_align_112 \
        --train_file_list $data_path/casia_align_train.list \
        --lfw_test_root $data_path/LFW/lfw_align_112 \
        --lfw_file_list $data_path/LFW/pairs.txt \
        --agedb_test_root $data_path/AgeDB-30/agedb30_align_112 \
        --agedb_file_list $data_path/AgeDB-30/agedb_30_pair.txt \
        --cfpfp_test_root $data_path/CFP-FP/CFP_FP_aligned_112 \
        --cfpfp_file_list $data_path/CFP-FP/cfp_fp_pair.txt \
        --total_epoch $train_epochs \
        --batch_size $batch_size \
        $PREC \
        --device_id ${ASCEND_DEVICE_ID} > ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done 
wait
sed -i "s|if total_iters == 700: break|if total_iters == 700: pass|g" train.py
#备份新生成的知识库文件
if [ -d "/usr/local/Ascend/opp/data/rl/Ascend910/custom" ]; then
    echo "Backup new rl autotune lib."
    mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/autotune/rl
    mv /usr/local/Ascend/opp/data/rl/Ascend910/custom ${cur_path}/test/output/$ASCEND_DEVICE_ID/autotune/rl
else
    echo "No RL autotune lib generated."
fi

if [ -d "/usr/local/Ascend/opp/data/tiling/ascend910/custom" ]; then
    echo "Backup new tiling autotune lib."
    mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/autotune/tiling
    mv /usr/local/Ascend/opp/data/tiling/ascend910/custom ${cur_path}/test/output/$ASCEND_DEVICE_ID/autotune/tiling
else
    echo "No tiling autotune lib generated."
fi

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#输出性能FPS，需要模型审视修改
time_per_iter=`grep "time:" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $8}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'/'${time_per_iter}'}'`
#输出训练精度,需要模型审视修改
train_accuracy=`grep "LFW Ave" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "Accuracy: " '{print $2}'`

#获取编译时间
CompileTime=`grep "step_time" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | head -1 | awk -F "step_time:" '{print $2}' | awk '{sum+=$1} END {print"",sum}' |sed s/[[:space:]]//g`

#打印，不需要修改
echo "------------------ Final result ------------------"
echo "Final Performance images/sec : $FPS"
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${time_per_iter}'*1000}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "loss:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk -F"," '{print $2}'|awk '{print $2}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "train_accuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CompileTime = ${CompileTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log