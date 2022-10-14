#!/bin/bash

cur_path=`pwd`/../
#失败用例打屏
export WORLD_SIZE=8
export JOB_ID=10087
RANK_ID_START=0

#基础参数，需要模型审视修改
#Batch Size
batch_size=256
#网络名称，同目录名称
Network="TransformerXL_RT2_ID4047_for_PyTorch"
#Device数量，单卡默认为1
RankSize=1
#训练epoch，可选
train_epochs=20
#训练step
train_steps=40000
#学习率
learning_rate=0.08

#参数配置
data_path=""

if [[ $1 == --help || $1 == --h ]];then
	echo "usage:./train_performance_1p.sh "
	exit 1
fi

for para in $*
do
	if [[ $para == --data_path* ]];then
		data_path=`echo ${para#*=}`
	fi
done

if [[ $data_path  == "" ]];then
	echo "[Error] para \"data_path\" must be config"
	exit 1
fi

#使能二进制
step_line=`grep "torch.npu.set_start_fuzz_compile_step(3)" ${cur_path}/pytorch/train.py -n | awk -F ':' '{print $1}'`
sed -i "${step_line}s/^/#/" ${cur_path}/pytorch/train.py
inc_line=`grep "torch.npu.global_step_inc()" ${cur_path}/pytorch/train.py -n | awk -F ':' '{print $1}'`
sed -i "${inc_line}s/^/#/" ${cur_path}/pytorch/train.py
line=`grep "import torch" ${cur_path}/pytorch/train.py -n | tail -1|awk -F ':' '{print $1}'`
sed -i "$[line+1]itorch.npu.set_compile_mode(jit_compile=False)" ${cur_path}/pytorch/train.py

##############执行训练##########
cd $cur_path
if [ -d $cur_path/test/output ];then
	rm -rf $cur_path/test/output/*
	mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
	mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

start=$(date +%s)
nohup python3 -m torch.distributed.launch \
	--nproc_per_node=8 \
	$cur_path/pytorch/train.py \
    --config_file pytorch/wt103_base.yaml \
    --config aiserver_1npu_fp32 \
	--affinity='disabled' \
	--work_dir=$cur_path/test/output/$ASCEND_DEVICE_ID \
	--batch_size=$batch_size \
    --batch_chunk=16 \
	--fp16 \
	--data=$data_path \
	--max_step=40000 \
	--multi_gpu=ddp \
	--lr=${learning_rate} > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2e_time=$(( $end - $start ))

###下面字段用于冒烟看护
BatchSize=${batch_size}
#设备类型，自动获取
DeviceType=`uname -m`
#用例名称，自动获取
CaseName=${Network}_bs${BatchSize}_${RankSize}'p'_'acc'

#结果打印，不需要修改
echo "-------------------- Final result --------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep "Training throughput" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk '{print$3}'`
ActualFPS=${FPS}
#打印，不需要修改
echo "Final performanceimages/sec: $ActualFPS"
echo "Final Training Duration sec : $e2e_time"
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`

#输出训练精度,需要模型审视修改
train_accuracy=`grep Eval $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| awk '{print$17}' | awk 'NR==1{min=$1;next}{min=min<$1?min:$1}END{print min}'`

#从train_$ASCEND_DEVICE_ID.log提取loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "ms/batch" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep -v "train loss"|awk -F 'loss ' '{print $2}'|awk -F '|' '{print $1}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print $1}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`


#关键信息打印到CaseName.log中，此处无需修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log