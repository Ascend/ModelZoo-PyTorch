source test/env_npu.sh
export RANK_SIZE=1

nohup python3 ./imagenet/main.py \
    --data /data/imagenet \
    --amp \
    --world-size 1 \
    --seed 60 \
    -a resnet50 \
    -j 64 \
    -b 512 \
    --lr 0.2 \
    --epochs 90 \
    --gpu 0 &