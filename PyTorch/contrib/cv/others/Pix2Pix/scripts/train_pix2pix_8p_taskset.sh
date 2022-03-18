
# source env_npu.sh
RANK_ID_START=0
export RANK_SIZE=8

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do

	KERNEL_NUM=$(($(nproc)/8))
	PID_START=$((KERNEL_NUM * RANK_ID))
	PID_END=$((PID_START + KERNEL_NUM - 1))

    export LOCAL_RANK=$RANK_ID
    export RANK=$RANK_ID

	nohup \
		taskset -c $PID_START-$PID_END python train.py \
    	--dataroot ./datasets/facades \
		--name facades_pix2pix_8p_bs1_lr0002_ep200 \
    	--model pix2pix \
		--direction BtoA \
		--gpu_ids 0,1,2,3,4,5,6,7 \
    	--norm instance \
		--display_freq 50 \
		--display_id -1  &
done
