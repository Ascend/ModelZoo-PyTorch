alias python="/usr/bin/python3.7.5"
source "./scripts/set_npu_env.sh"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
nohup python3.7.5 -u main_npu_8p.py \
	"/home/data/imagenet/" \
	      --addr=$(hostname -I |awk '{print $1}') \
        --lr=0.4 \
        --print-freq=10 \
        --wd=0.0005 \
        --workers=$(nproc) \
        --epochs=90 \
        --amp \
        --world-size=1 \
        --dist-backend='hccl' \
        --multiprocessing-distributed \
        --loss-scale=128.0 \
        --opt-level='O2' \
        --device='npu' \
        --rank=0 \
        --warm_up_epochs=5 \
        --batch-size=2048 > ./wide_resnet101_2_npu_8p.log 2>&1 &
