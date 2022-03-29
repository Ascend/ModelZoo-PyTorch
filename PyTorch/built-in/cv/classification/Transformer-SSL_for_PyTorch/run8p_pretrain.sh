source env_npu.sh

export WORLD_SIZE=8
for i in $(seq 0 7)
do
    export RANK=$i
    start=$((24 * i))
    end=$((start + 23))
    taskset -c $start-$end nohup python -u moby_main.py \
            --cfg configs/moby_swin_tiny.yaml \
            --data-path /data/imagenet \
            --local_rank $i \
            --batch-size 128 > train_${i}.log 2>&1 &
done