alias python="/usr/bin/python3.7.5"
source "/home/wide_resnet101_2/scripts/set_npu_env.sh"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
nohup python3.7.5 -u main_npu_8p.py \
	"/home/data/imagenet/" \
	      --addr=$(hostname -I |awk '{print $1}') \
        --workers=$(nproc) \
        --evaluate \
        --print-freq=1 \
        --world-size=1 \
        --pretrained \
        --dist-backend='hccl' \
        --device='npu' \
        --multiprocessing-distributed \
        --rank=0 \
        --batch-size=2048 > ./wide_resnet101_2_npu_8p_eval.log 2>&1 &