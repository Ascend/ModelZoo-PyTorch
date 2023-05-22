
export PYTHONPATH=./vision-transformers-cifar10:$PYTHONPATH

WORLD_SIZE=4
RANK_ID_START=0

KERNEL_NUM=$(($(nproc) / 8))
for ((RANK_ID = $RANK_ID_START; RANK_ID < $((WORLD_SIZE + RANK_ID_START)); RANK_ID++))
do
	PID_START=$((KERNEL_NUM * $RANK_ID))
	PID_END=$((PID_START + KERNEL_NUM - 1))

	taskset -c $PID_START-$PID_END nohup python -u train_cifar10.py --net swin --bs 512 --lr 4e-4 --n_epoch 400 --local_rank $RANK_ID --world_size $WORLD_SIZE --noamp &

done
