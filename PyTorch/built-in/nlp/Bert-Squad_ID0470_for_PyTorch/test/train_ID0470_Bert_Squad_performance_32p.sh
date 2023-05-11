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
export RANK_SIZE=32
RANK_ID_START=0

#非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source  ${test_path_dir}/env_npu.sh
fi

# 数据集路径,保持为空,不需要修改
data_path=""
ckpt_path=""
conf_path=""
server_index=""
fix_node_ip=""
devicesnum=""


#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="Bert-Squad_ID0470_for_PyTorch"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=32
#训练step
train_steps=
#学习率
learning_rate=2e-4


#维测参数，precision_mode需要模型审视修改
precision_mode="allow_fp32_to_fp16"


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
    --batch_size                train batch size
    -h/--help		             show help message
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
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
	elif [[ $para == --fix_node_ip* ]];then
	    fix_node_ip=`echo ${para#*=}`
	elif [[ $para == --devicesnum* ]];then
	    devicesnum=`echo ${para#*=}`
    elif [[ $para == --conf_path* ]];then
            conf_path=`echo ${para#*=}`
    elif [[ $para == --server_index* ]];then
            server_index=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    fi
done

one_node_ip=`find $conf_path -name "server_*0.info"|awk -F "server_" '{print $2}'|awk -F "_" '{print $1}'`
linux_num=`find $conf_path -name "server_*.info" |wc -l`

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#训练开始时间，不需要修改



export HCCL_IF_IP=$fix_node_ip
export MASTER_ADDR=$one_node_ip
export MASTER_PORT=29668
export HCCL_WHITELIST_DISABLE=1
device_num=${#devicesnum}
devices_num=`awk 'BEGIN{printf "%.0f\n",'${device_num}'-1}'`

start_time=$(date +%s)
NPUS=($(seq 0 $devices_num))
rank_server=`awk 'BEGIN{printf "%.0f\n",'${device_num}'*'${server_index}'}'`
export NPU_WORLD_SIZE=`awk 'BEGIN{printf "%.0f\n",'${device_num}'*'${linux_num}'}'`


#进入训练脚本目录，需要模型审视修改
cd $cur_path
mkdir -p results/SQUAD
rank=0
for((RANK_ID=$RANK_ID_START;RANK_ID<8;RANK_ID++));
do
    #设置环境变量，不需要修改
    export ASCEND_DEVICE_ID=$RANK_ID
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID
    export RANK=`awk 'BEGIN{printf "%.0f\n",'${rank}'+'${rank_server}'}'`


    #创建DeviceID输出目录，不需要修改
    if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID/ckpt
    fi

    
    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path
    nohup python3 run_squad.py \
          --init_checkpoint ${ckpt_path}/bert_large_pretrained_amp.pt \
          --bert_model bert-large-uncased \
		  --do_train \
		  --train_file ${data_path}/train-v1.1-min.json \
		  --train_batch_size ${batch_size} \
		  --do_predict \
		  --predict_batch_size ${batch_size} \
		  --predict_file ${data_path}/dev-v1.1.json \
		  --learning_rate ${learning_rate} \
		  --num_train_epochs ${train_epochs} \
		  --seed 1 \
		  --fp16 \
		  --max_steps 100 \
		  --use_npu \
		  --loss_scale 4096 \
		  --vocab_file ${data_path}/data/uncased_L-24_H-1024_A-16/vocab.txt \
		  --do_eval \
          --eval_script ${data_path}/evaluate-v1.1.py \
		  --npu_id ${ASCEND_DEVICE_ID} \
		  --do_lower_case \
		  --output_dir results/SQUAD \
		  --config_file bert_config.json \
		  --num_npu 32 \
		  --local_rank=$RANK_ID \
          --addr $one_node_ip \
		  --json-summary ${test_path_dir}/output/${ASCEND_DEVICE_ID}/dllogger.json> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
		  let rank++
done 
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
step_time=`grep 'step_time : ' $test_path_dir/output/0/train_0.log| awk '{print$13}'| tail -n +3 |awk '{sum+=$1} END {print"",sum/NR}' | sed s/[[:space:]]//g`

FPS=`awk 'BEGIN{printf "%.2f\n", '$batch_size'/'$step_time'*32}'`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep 'F1 : ' $test_path_dir/output/0/train_0.log|awk '{print $10}'`
#打印，不需要修改
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
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep -r "step_loss :" $test_path_dir/output/0/train_0.log | awk -F 'step_loss : ' '{print $2}'|awk '{print $1}' > $test_path_dir/output/0/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/0/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $test_path_dir/output/0/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $test_path_dir/output/0/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $test_path_dir/output/0/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $test_path_dir/output/0/${CaseName}.log
echo "CaseName = ${CaseName}" >> $test_path_dir/output/0/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $test_path_dir/output/0/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $test_path_dir/output/0/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $test_path_dir/output/0/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $test_path_dir/output/0/${CaseName}.log
