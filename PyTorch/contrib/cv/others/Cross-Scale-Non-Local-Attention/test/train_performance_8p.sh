data_path="/home/CSNLN"
for para in $*
do
    if [[ $para == --data_path* ]]; then
        data_path=`echo ${para#*=}`
    fi
done
export RANK_SIZE=8
for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++));
do
    KERNEL_NUM=8
    PID_START=$((KERNEL_NUM * RANK_ID))
    PID_END=$((PID_START + KERNEL_NUM - 1))
    export RANK_ID=$RANK_ID
    nohup taskset -c $PID_START-$PID_END python3 main.py \
        --epochs 1000 \
	    --model CSNLN \
	    --data_test Set5 \
	    --dir_data ${data_path} \
	    --scale 2 \
	    --n_feats 128 \
	    --depth 12 \
	    --rank_id $RANK_ID \
	    --chop \
	    --batch_size 16 \
	    --patch_size 96 \
	    --save CSNLN_x2 \
	    --data_train DIV2K \
	    --save_models \
	    --n_threads 18 \
	    --distributed 1 \
	    --n_GPUs 8 \
	    --print_every 1 \
	    --seed 12 \
	    --amp \
	    --lr 0.0001 \
	    --performance &
done
