#!/bin/bash

cur_path=`pwd`/../
#失败用例打屏
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SCALAR_TO_HOST_MEM=1

export MKL_SERVICE_FORCE_INTEL=1
export BMMV2_ENABLE=1
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
#基础参数，需要模型审视修改
#Batch Size
batch_size=512
#网络名称，同目录名称
Network="mBART_ID2372_for_PyTorch"
#Device数量，单卡默认为1
RankSize=1
#训练epoch，可选
train_epochs=1
#训练step
train_steps=
#学习率
learning_rate=3e-05

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
    elif [[ $para == --conda_name* ]];then
        conda_name=`echo ${para#*=}`
        source set_conda.sh --conda_name=$conda_name
        #export PATH=/usr/local/python3.7.5/bin:/home/anaconda3/bin:$PATH
        #source activate py8
        source activate $conda_name
        
	fi
done

if [[ $data_path  == "" ]];then
	echo "[Error] para \"data_path\" must be config"
	exit 1

fi
sed -i "s|checkpoint_utils.save_checkpoint(|#checkpoint_utils.save_checkpoint(|g" $cur_path/fairseq_cli/train.py
##############执行训练##########
cd $cur_path
if [ -d $cur_path/test/output ];then
	rm -rf $cur_path/test/output/*
	mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
	mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait


pip3 install --editable ./
start=$(date +%s)
python3 train.py $data_path/en_ro/ \
  --distributed-world-size 1 --npu --npu-id $ASCEND_DEVICE_ID --fp16 --encoder-normalize-before --decoder-normalize-before \
  --arch mbart_large --layernorm-embedding \
  --task translation_from_pretrained_bart \
  --source-lang en_XX --target-lang ro_RO \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 --warmup-updates 2500 --total-num-update 40000 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 512 --update-freq 2 \
  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --seed 222 --log-format simple --log-interval 2 \
  --restore-file $data_path/mbart.cc25/model.pt \
  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
  --langs $langs \
  --max-epoch $train_epochs \
  --max-update 200 \
  --ddp-backend no_c10d > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

sed -i "s|#checkpoint_utils.save_checkpoint(|checkpoint_utils.save_checkpoint(|g" $cur_path/fairseq_cli/train.py
#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=0
FPS=`grep -rn train_inner $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F "wps=" '{print$2}' | awk -F "," '{print$1}' | tail -n+6 | awk '{sum+=$1} END {print"",sum/NR}' | sed s/[[:space:]]//g`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*1000/'${FPS}'}'`
#输出训练精度,需要模型审视修改
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"
#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep train_inner $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F "loss=" '{print$2}' | awk -F "," '{print$1}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log

