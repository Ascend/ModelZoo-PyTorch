source ./test/env_npu.sh
export RANK_SIZE=1
PORT=29500 ./tools/dist_train.sh configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes_1p.py 1  \
                                                      --device npu --seed 1 --deterministic