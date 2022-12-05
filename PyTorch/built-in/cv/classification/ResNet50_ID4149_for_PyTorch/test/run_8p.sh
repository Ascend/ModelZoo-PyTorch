for((RANK_ID=0;RANK_ID<8;RANK_ID++));
do
    source test/env_npu.sh
    export RANK_SIZE=8
    export RANK_ID=$RANK_ID
    nohup python3 ./imagenet/main.py \
        --data /data/imagenet \
        --amp \
        --world-size 1 \
        --seed 60 \
        -a resnet50 \
        -j 128 \
        -b 4096 \
        --lr 1.6 \
        --epochs 90 \
        --gpu ${RANK_ID} \
        --rank 0 \
        --multiprocessing-distributed &
done