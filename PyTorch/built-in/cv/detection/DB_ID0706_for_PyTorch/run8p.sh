source env.sh
export PYTHONPATH=./:$PYTHONPATH
export TASK_QUEUE_ENABLE=0
export DYNAMIC_OP="ADD"

kernel_num=$(nproc)

if [ ${kernel_num} -lt 95 ];then
    cpu_number=${kernel_num}
else
    cpu_number=95
fi

taskset -c 0-${cpu_number} python3.7 -W ignore train.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume path-to-model-directory/MLT-Pretrain-ResNet50 \
        --data_path datasets/icdar2015/ \
        --seed=515 \
        --distributed \
        --device_list "0,1,2,3,4,5,6,7" \
        --num_gpus 8 \
        --local_rank 0 \
        --dist_backend 'hccl' \
        --world_size 1 \
        --batch_size 128 \
        --lr 0.056 \
        --addr $(hostname -I |awk '{print $1}') \
        --amp \
        --Port 29502 \