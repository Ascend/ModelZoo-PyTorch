source scripts/set_env.sh
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
python3 train.py \
    --device npu \
    --device_num 8 \
    --world_size 1 \
    --batchSz 4 \
    --seed 1024 \
    --lr 1e-3 \
    --lr_decay 0.3 \
    --data /opt/npu/dataset/luna16 \
    --save model_8p_scripts \
    --distributed \
    --dist_backend hccl \
    --amp \
    --opt_level O2 \
    --loss_scale 128