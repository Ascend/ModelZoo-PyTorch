source ../../test/env.sh
export RANK_SIZE=8
for((RANK_ID=0;RANK_ID<8;RANK_ID++));
do
    export RANK_ID=$RANK_ID

    nohup python3 train.py  \
        --model googlenet \
        --epochs 2 \
        --data-path=/data/imagenet \
        --distributed \
        --batch-size 4096 \
        --workers 128 \
        --lr 0.8 \
        --momentum 0.9 \
        --apex \
        --apex-opt-level O1 \
        --loss_scale_value 1024 \
        --weight-decay 1e-4 \
        --print-freq 1 &
done