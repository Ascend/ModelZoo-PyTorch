source env.sh
export PYTHONPATH=./:$PYTHONPATH
export TASK_QUEUE_ENABLE=1
export DYNAMIC_OP="ADD"
taskset -c 0-23 python3.7 train.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml \
        --resume path-to-model-directory/MLT-Pretrain-ResNet50 \
        --data_path datasets/icdar2015/ \
        --seed=515 \
        --amp \
        --device_list "0"
