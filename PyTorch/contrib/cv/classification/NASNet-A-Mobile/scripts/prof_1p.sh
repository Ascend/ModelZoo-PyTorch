alias python="/usr/bin/python3.7.5"
source scripts/env_npu.sh
nohup python3 -u main_npu_1p.py \
	"/home/data/imagenet/" \
	--prof \
        --lr=1.0 \
        --print-freq=10 \
        --momentum=0.9 \
        --epochs=1 \
        --workers=$(nproc) \
        --seed=49 \
        --amp \
        --loss-scale=128.0 \
        --opt-level='O2' \
        --device='npu' \
        --world-size=1 \
        --batch-size=256 > ./log/nasnet_mobile_npu_1p_bs256.log 2>&1 &