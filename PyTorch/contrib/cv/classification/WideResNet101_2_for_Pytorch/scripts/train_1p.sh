alias python="/usr/bin/python3.7.5"
source "./scripts/set_npu_env.sh"
nohup python3.7.5 -u main_npu_1p.py \
	"/home/data/imagenet/" \
        --lr=0.2 \
        --print-freq=10 \
        --epochs=1 \
        --amp \
        --loss-scale=128.0 \
        --opt-level='O2' \
        --device='npu' \
        --world-size=1 \
        --batch-size=256 > ./wide_resnet101_2_npu_1p.log 2>&1 &


