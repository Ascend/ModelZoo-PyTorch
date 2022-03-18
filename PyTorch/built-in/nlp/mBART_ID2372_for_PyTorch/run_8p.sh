source env.sh
PRETRAIN=/path/model.pt                 # fix it to your own model path
DATA_PATH=path_of_data                   # fix it to your own data path
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN


export RANK_SIZE=8
for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
do
    export RANK=$RANK_ID
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
							  --max-tokens 512 --update-freq 2 \
							  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
							  --seed 222 --log-format simple --log-interval 2 \
							  --restore-file $PRETRAIN \
							  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
							  --langs $langs \
							  --ddp-backend no_c10d &
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
							  --max-tokens 512 --update-freq 2 \
							  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
							  --seed 222 --log-format simple --log-interval 2 \
							  --restore-file $PRETRAIN \
							  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
							  --langs $langs \
							  --ddp-backend no_c10d &
	fi
done
