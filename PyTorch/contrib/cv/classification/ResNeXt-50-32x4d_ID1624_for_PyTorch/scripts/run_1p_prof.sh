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
        --gpu=0 \
        --world-size=1 \
        --epochs=1\
        --amp \
        --batch-size=256 \
        --prof > ./resNext50_1p_prof.log 2>&1