source ../../test/env.sh

nohup python3 train.py  \
        --model googlenet \
        --epochs 2 \
        --apex \
        --apex-opt-level O1 \
        --loss_scale_value 1024 \
        --seed 1234 \
        --data-path=/home/ImageNet2012 \
        --batch-size 512 \
        --workers 16 \
        --lr 1e-2 \
        --momentum 0.9 \
        --weight-decay 1e-4 \
        --print-freq 10 &
