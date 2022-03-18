pkill python3
export HCCL_WHITELIST_DISABLE=1
export WORLD_SIZE=1
export RANK=0


for i in $(seq 0 0)
do
    python3 train.py train.yaml \
	    --distributed_launch \
		--distributed_backend=hccl \
        --local_rank ${i} &
done

wait