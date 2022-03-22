#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`
source env.sh

#集合通信参数,不需要修改

export RANK_SIZE=8

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="mBART_for_PyTorch"
#训练batch_size
token_size=1024

#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../

# 将对应的数据以及模型等放到对应路径 或 修改以下路径以适应本地训练
DATA_PATH=train_data/en_ro
PRETRAIN=mbart.cc25/model.pt
BPE_PATH=mbart.cc25/sentence.bpe.model
model_dir=checkpoints/checkpoint_best.pt
SCRIPTS=mosesdecoder/scripts
WMT16_SCRIPTS=wmt16-scripts

REPLACE_UNICODE_PUNCT=$SCRIPTS/tokenizer/replace-unicode-punctuation.perl
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
NORMALIZE_ROMANIAN=$WMT16_SCRIPTS/preprocess/normalise-romanian.py
REMOVE_DIACRITICS=$WMT16_SCRIPTS/preprocess/remove-diacritics.py
HYP=hyp
REF=ref
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
#创建DeviceID输出目录，不需要修改


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
			taskset -c $a-$b fairseq-train $DATA_PATH --fp16 --distributed-world-size 8 --npu \
							  --device-id $RANK_ID --distributed-rank $RANK_ID --distributed-no-spawn --max-update 40000 \
							  --encoder-normalize-before --decoder-normalize-before \
							  --arch mbart_large --layernorm-embedding \
							  --task translation_from_pretrained_bart \
							  --source-lang en_XX --target-lang ro_RO \
							  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
							  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
							  --lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 --warmup-updates 2500 --total-num-update 40000 \
							  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
							  --max-tokens 1024 --update-freq 2 \
							  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
							  --seed 222 --log-format simple --log-interval 2 \
							  --restore-file $PRETRAIN \
							  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
							  --langs $langs \
							  --ddp-backend no_c10d > ${cur_path}/output/${RANK_ID}/train_${RANK_ID}.log 2>&1 &
	else
		fairseq-train $DATA_PATH --fp16 --distributed-world-size 8 --npu \
							  --device-id $RANK_ID --distributed-rank $RANK_ID --distributed-no-spawn --max-update 40000 \
							  --encoder-normalize-before --decoder-normalize-before \
							  --arch mbart_large --layernorm-embedding \
							  --task translation_from_pretrained_bart \
							  --source-lang en_XX --target-lang ro_RO \
							  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
							  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
							  --lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 --warmup-updates 2500 --total-num-update 40000 \
							  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
							  --max-tokens 1024 --update-freq 2 \
							  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
							  --seed 222 --log-format simple --log-interval 2 \
							  --restore-file $PRETRAIN \
							  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
							  --langs $langs \
							  --ddp-backend no_c10d > ${cur_path}/output/${RANK_ID}/train_${RANK_ID}.log 2>&1 &
	fi
done
wait
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))


fairseq-generate $DATA_PATH \
  --fp16 --path $model_dir --max-tokens 4096 \
  --task translation_from_pretrained_bart \
  --gen-subset test \
  -t ro_RO -s en_XX \
  --bpe 'sentencepiece' --sentencepiece-model $BPE_PATH \
  --scoring sacrebleu --remove-bpe 'sentencepiece' \
  --batch-size 32 --langs $langs > en_ro
sed -i '$d' en_ro
cat en_ro | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[ro_RO\]//g' |$TOKENIZER ro > $HYP".txt"
cat en_ro | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[ro_RO\]//g' |$TOKENIZER ro > $REF".txt"

for f in $HYP $REF
	do
	rm -rf "en_ro."$f
	cat $f".txt" | \
	perl $REPLACE_UNICODE_PUNCT | \
	perl $NORM_PUNC -l ro | \
	perl $REM_NON_PRINT_CHAR | \
	python3 $NORMALIZE_ROMANIAN | \
	python3 $REMOVE_DIACRITICS | \
	perl $TOKENIZER -no-escape -threads 16 -a -l ro >"en_ro."$f
	done
sacrebleu -tok 'none' -s 'none' en_ro.ref < en_ro.hyp > res.log
wait
ASCEND_DEVICE_ID=0


#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
WPS=`grep 'train_inner ' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "wps=" '{print $NF}'|awk -F "wps" '{print $1}'|awk -F "," '{print $1}'|awk 'END {print}'`
train_wall=`grep 'train_inner ' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F "train_wall=" '{print $NF}'|awk 'NR==1{min=$1;next}{min=min<$1?min:$1}END{print min}'`
#打印，不需要修改
echo "Final Performance images/sec : $WPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep 'version.1.5.1 = ' res.log |awk '{print $3}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
TokenSize=${token_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${TokenSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualWPS=${WPS}
#单迭代训练时长
TrainingTime=${train_wall}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep -r "step_loss :" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print $19}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TokenSize = ${TokenSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualWPS = ${ActualWPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}">> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log