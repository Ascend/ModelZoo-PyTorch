source ../../test/env_npu.sh
rm -f nohup.out

nohup python3 train.py  \
        --model densenet161 \
        --epochs 90 \
        --data-path=/data/imagenet \
        --batch-size 128 \
        --workers 16 \
        --lr 0.1 \
        --momentum 0.9 \
        --apex \
        --apex-opt-level O2 \
        --loss_scale_value dynamic \
        --weight-decay 1e-4 \
        --print-freq 1 &
