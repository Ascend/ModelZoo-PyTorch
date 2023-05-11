#!/bin/bash

################基础配置参数，需要模型审视修改##################
# 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
# 网络名称，同目录名称
Network="Speech-Transformer"
shuffle=1
# 训练batch_size
batch_size=128
batch_frames=0
maxlen_in=800
maxlen_out=150
# 训练使用的npu卡数
export RANK_SIZE=8
# 数据集路径,保持为空,不需要修改
data_path=""

dumpdir=dump   # directory to dump full features

# Feature configuration
do_delta=false
LFR_m=4  # Low Frame Rate: number of frames to stack
LFR_n=3  # Low Frame Rate: number of frames to skip

# Network architecture
# Encoder
d_input=80
n_layers_enc=6
n_head=8
d_k=64
d_v=64
d_model=512
d_inner=2048
dropout=0.1
pe_maxlen=5000
# Decoder
d_word_vec=512
n_layers_dec=6
tgt_emb_prj_weight_sharing=1
# Loss
label_smoothing=0.1

# Training config
epochs=150

# optimizer
k=0.4
warmup_steps=4000
# save & logging
checkpoint=0
continue_from=""
print_freq=1
visdom=0
visdom_lr=0
visdom_epoch=0
visdom_id="Transformer Training"


source path.sh

feat_train_dir=${dumpdir}/train/delta${do_delta};
feat_test_dir=${dumpdir}/test/delta${do_delta};
feat_dev_dir=${dumpdir}/dev/delta${do_delta};
dict=data/lang_1char/train_chars.txt
echo "dictionary: ${dict}"


# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

# 校验是否传入data_path,不需要修改
# if [[ $data_path == "" ]];then
#     echo "[Error] para \"data_path\" must be confing"
#     exit 1
# fi


###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径

cur_path=`pwd`
cur_path_last_diename=${cur_path##*/}
if [ x"${cur_path_last_diename}" == x"test" ];then
    test_path_dir=${cur_path}
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi
echo $test_path_dir

#################创建日志输出目录，不需要修改#################
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi


#################启动训练脚本#################
#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

export WORLD_SIZE=$RANK_SIZE
expdir=${test_path_dir}/output


RANK_ID_START=0
RANK_SIZE=8
echo "Network Training"
for((rank_id=$RANK_ID_START;rank_id<$((RANK_SIZE+RANK_ID_START));rank_id++));
do
KERNEL_NUM=$(($(nproc)/$RANK_SIZE))
PID_START=$((KERNEL_NUM * rank_id))
PID_END=$((PID_START + KERNEL_NUM - 1))
taskset -c $PID_START-$PID_END \
    python3 \
    ../src/bin/train.py \
    --train-json ${feat_train_dir}/data.json \
    --valid-json ${feat_dev_dir}/data.json \
    --dict ${dict} \
    --LFR_m ${LFR_m} \
    --LFR_n ${LFR_n} \
    --d_input $d_input \
    --n_layers_enc $n_layers_enc \
    --n_head $n_head \
    --d_k $d_k \
    --d_v $d_v \
    --d_model $d_model \
    --d_inner $d_inner \
    --dropout $dropout \
    --pe_maxlen $pe_maxlen \
    --d_word_vec $d_word_vec \
    --n_layers_dec $n_layers_dec \
    --tgt_emb_prj_weight_sharing $tgt_emb_prj_weight_sharing \
    --label_smoothing ${label_smoothing} \
    --epochs $epochs \
    --shuffle $shuffle \
    --batch-size $batch_size \
    --batch_frames $batch_frames \
    --maxlen-in $maxlen_in \
    --maxlen-out $maxlen_out \
    --k $k \
    --num-workers $KERNEL_NUM \
    --warmup_steps $warmup_steps \
    --save-folder ${expdir} \
    --checkpoint $checkpoint \
    --continue-from "$continue_from" \
    --print-freq ${print_freq} \
    --visdom $visdom \
    --visdom_lr $visdom_lr \
    --visdom_epoch $visdom_epoch \
    --local_rank ${rank_id} \
    --is_distributed True \
    --visdom-id "$visdom_id" \
    >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done

wait

##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a FPS ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | grep -v cross | tail -n 5 | awk '{print $NF}' | awk '{sum+=$1} END {print sum/NR}'` 
#打印，不需要修改
echo "Final Performance sentences/sec : $FPS"

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
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`
#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep 'Train Loss' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{ print $14 }' | awk 'END { print }' >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

# 最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}'  ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
