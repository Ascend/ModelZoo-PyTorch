#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export SCALAR_TO_HOST_MEM=1

export MKL_SERVICE_FORCE_INTEL=1
export BMMV2_ENABLE=1
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

#集合通信参数,不需要修改
export RANK_SIZE=8
train_epochs=1
#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="mBART_ID2372_for_PyTorch"
#训练batch_size
token_size=512

#训练开始时间，不需要修改
start_time=$(date +%s)
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

sed -i "s|checkpoint_utils.save_checkpoint(|#checkpoint_utils.save_checkpoint(|g" $cur_path/../fairseq_cli/train.py

##############执行训练##########
cd $cur_path/../
pip3 install --editable ./
#进入训练脚本目录，需要模型审视修改

export RANK_SIZE=8
for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
do
    export RANK=$RANK_ID
    if [ -d ${cur_path}/output/${RANK_ID} ];then
        rm -rf ${cur_path}/output/${RANK_ID}
        mkdir -p ${cur_path}/output/$RANK_ID
      else
            mkdir -p ${cur_path}/output/$RANK_ID
    fi
	if [ $(uname -m) = 'aarch64' ]
		then
			let a=0+RANK_ID*24
			let b=23+RANK_ID*24
			taskset -c $a-$b python3 train.py $data_path/en_ro/ --fp16 --distributed-world-size 8 --npu \
							  --device-id $RANK_ID --distributed-rank $RANK_ID --distributed-no-spawn --max-update 50 \
							  --encoder-normalize-before --decoder-normalize-before \
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
                              --max-epoch $train_epochs \
							  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
							  --langs $langs \
							  --ddp-backend no_c10d > ${cur_path}/output/${RANK_ID}/train_${RANK_ID}.log 2>&1 &
	else
		python3 train.py $data_path/en_ro/ --fp16 --distributed-world-size 8 --npu \
							  --device-id $RANK_ID --distributed-rank $RANK_ID --distributed-no-spawn --max-update 50 \
							  --encoder-normalize-before --decoder-normalize-before \
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
							  --ddp-backend no_c10d > ${cur_path}/output/${RANK_ID}/train_${RANK_ID}.log 2>&1 &
	fi
done
wait

ASCEND_DEVICE_ID=0
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))


#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能WPS，需要模型审视修改
WPS=`grep 'train_inner ' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "wps=" '{print $NF}'|awk -F "wps" '{print $1}'|awk -F "," '{print $1}'|awk 'END {print}'`
train_wall=`grep 'train_inner ' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "train_wall=" '{print $NF}'|awk 'NR==1{min=$1;next}{min=min<$1?min:$1}END{print min}'|awk -F "," '{print$1}'`
#打印，不需要修改
echo "Final Performance images/sec : $WPS"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
TokenSize=${token_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${TokenSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualWPS=${WPS}
#单迭代训练时长
TrainingTime=${train_wall}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep -r "loss=" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F "loss=" '{print $2}' |awk -F "," '{print $1}'  > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${TokenSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualWPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log

sed -i "s|#checkpoint_utils.save_checkpoint(|checkpoint_utils.save_checkpoint(|g" $cur_path/../fairseq_cli/train.py