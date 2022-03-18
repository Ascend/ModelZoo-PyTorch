source ./test/env_npu.sh
export RANK_SIZE=8
rm -f nohup.out

for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++));
do
    export RANK_ID=$RANK_ID

    nohup python3 train.py  \
        --model densenet201 \
        --epochs 40 \
        --data-path=/data/imagenet \
        --distributed \
        --batch-size 1024 \
        --workers 128 \
        --lr 0.8 \
        --momentum 0.9 \
        --apex \
        --apex-opt-level O2 \
        --loss_scale_value 1024 \
        --weight-decay 1e-4 \
        --print-freq 1 &
done