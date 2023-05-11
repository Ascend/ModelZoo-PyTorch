source scripts/set_env.sh

python3 eval.py \
    --batchSz 4 \
    --data /opt/npu/dataset/luna16 \
    --device npu \
    --amp \
    --resume vnet_model_best.pth.tar \