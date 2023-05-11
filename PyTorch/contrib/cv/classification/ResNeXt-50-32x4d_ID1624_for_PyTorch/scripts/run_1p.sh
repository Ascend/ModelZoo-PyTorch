source scripts/set_npu_env.sh
python3 ./main.py \
	/opt/npu/imagenet/ \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49 \
        --workers=128 \
        --learning-rate=0.1 \
        --mom=0.9 \
        --weight-decay=1.0e-04  \
        --print-freq=1 \
        --dist-backend 'hccl' \
        --rank=0 \
        --device='npu' \
        --world-size=1 \
        --gpu=0 \
        --epochs=1\
        --amp \
        --batch-size=256 > ./resnext50_1p.log 2>&1