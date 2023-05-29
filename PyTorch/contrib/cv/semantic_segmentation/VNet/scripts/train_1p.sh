source scripts/set_env.sh

python3 train.py \
    --device npu \
    --device_num 1 \
    --world_size 1 \
    --batchSz 8 \
    --lr 1e-4 \
    --data /opt/npu/dataset/luna16 \
    --save model_1p_scripts \
    --amp \
    --opt_level O2 \
    --loss_scale 128