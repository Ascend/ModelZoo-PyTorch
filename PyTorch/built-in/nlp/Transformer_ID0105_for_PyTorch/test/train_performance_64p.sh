#!/bin/bash

#当前路径,不需要修改
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
export RANK_SIZE=64
export MASTER_PORT=29688
export HCCL_WHITELIST_DISABLE=1
export BMMV2_ENABLE=1
# 数据集路径,保持为空,不需要修改
data_path=""
conf_path=""
server_index=""
fix_node_ip=""
one_node_ip=""
linux_num=""

#网络名称,同目录名称,需要模型审视修改
Network="Transformer_ID0105_for_PyTorch"

#训练batch_size,,需要模型审视修改
batch_size=128



# 参数校验，data_path为必传参数，其他参数的增删由模型自身决定；此处新增参数需在上面有定义并赋值
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --epochs* ]];then
        epochs=`echo ${para#*=}`
    elif [[ $para == --conf_path* ]];then
            conf_path=`echo ${para#*=}`
    elif [[ $para == --server_index* ]];then
            server_index=`echo ${para#*=}`
    elif [[ $para == --fix_node_ip* ]];then
            fix_node_ip=`echo ${para#*=}`
    elif [[ $para == --one_node_ip* ]];then
            one_node_ip=`echo ${para#*=}`
    elif [[ $para == --linux_num* ]];then
            linux_num=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

if [[ $conf_path == "" ]];then
    one_node_ip=$one_node_ip
    linux_num=$linux_num
else
    one_node_ip=`find $conf_path -name "server_*0.info"|awk -F "server_" '{print $2}'|awk -F "_" '{print $1}'`
    linux_num=`find $conf_path -name "server_*.info" |wc -l`
fi

export HCCL_IF_IP=$fix_node_ip
export MASTER_ADDR=$one_node_ip

ASCEND_DEVICE_ID=0
#创建DeviceID输出目录，不需要修改
if [ -d $cur_path/output/${ASCEND_DEVICE_ID} ];then
    rm -rf $cur_path/output/$ASCEND_DEVICE_ID
    mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
else
    mkdir -p $cur_path/output/$ASCEND_DEVICE_ID
fi

#################启动训练脚本#################

# 非平台场景时source 环境变量
#check_etp_flag=`env | grep etp_running_flag`
#etp_flag=`echo ${check_etp_flag#*=}`
#if [ x"${etp_flag}" != x"true" ];then
#    source ${test_path_dir}/env_npu.sh
#fi

# 必要参数替换配置文件
cd $cur_path
DATA_DIR=./data/dataset/wmt14_en_de_joined_dict/
MODELDIR="./checkpoints/"
mkdir -p "$MODELDIR"
LOGFILE="$MODELDIR/log"
STAT_FILE="log.txt"

sed -i "s|if i>100:pass|if i>100:break|g" train_8p.py
sed -i "s|if m >=2:pass|if m >=2:break|g" train_8p.py

export ASCEND_GLOBAL_LOG_LEVEL_ETP=3
export PTCOPY_ENABLE=1
export TASK_QUEUE_ENABLE=1
export DYNAMIC_OP="ADD#MUL"
start_time=$(date +%s)
NPUS=($(seq 0 7))
rank_server=`awk 'BEGIN{printf "%.0f\n",8*'${server_index}'}'`
export NPU_WORLD_SIZE=`awk 'BEGIN{printf "%.0f\n",8*'${linux_num}'}'`
rank=0
for i in ${NPUS[@]}
do
    export NPU_CALCULATE_DEVICE=${i}
    mkdir -p  $test_path_dir/output/${i}/
    export ASCEND_DEVICE_ID=${i}
    export RANK=`awk 'BEGIN{printf "%.0f\n",'${rank}'+'${rank_server}'}'`
    echo run process ${rank}


    nohup python3 train_8p.py \
       $data_path \
      --arch transformer_wmt_en_de \
      --share-all-embeddings \
      --optimizer adam \
  --adam-beta1 0.9 \
  --distributed-world-size ${NPU_WORLD_SIZE} \
  --adam-beta2 0.997 \
  --addr ${one_node_ip} \
  --port 29990 \
  --adam-eps "1e-9" \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 0.0 \
  --warmup-updates 4000 \
  --lr 0.0006 \
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion cross_entropy \
  --label-smoothing 0.1 \
  --max-sentences 128\
  --max-tokens 102400 \
  --seed 1 \
  --save-dir $MODELDIR \
  --stat-file $STAT_FILE\
  --log-interval 1\
  --amp\
  --device-id ${rank}\
  --amp-level O2  >  $test_path_dir/output/${i}/train_${i}.log 2>&1 &
    let rank++
done
wait
sed -i "s|if i>100:break|if i>100:pass|g" train_8p.py
sed -i "s|if m >=2:break|if m >=2:pass|g" train_8p.py



##################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
time=` grep -rns "Time" $test_path_dir/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log |grep -v "all" |awk -F "Time" '{print$2}' |awk -F "(" '{print$1}'|tail -n +5|awk '{sum+=$1} END {print"",64*128*NR/sum}'|sed s/[[:space:]]//g`
FPS=`python3 -c "print(${time}*96)"`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

#获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep -rns "Time" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log |grep -v "all" |awk -F "Loss" '{print$2}' |awk -F "(" '{print$1}'  > $test_path_dir/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
