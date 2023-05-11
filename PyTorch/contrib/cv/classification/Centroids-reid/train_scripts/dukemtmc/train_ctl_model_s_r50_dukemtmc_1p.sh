source env_npu.sh

export RANK_SIZE=1
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

python3 train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'DukeMTMC-reID' \
DATASETS.ROOT_DIR '/home/xyc/data/' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs/dukemtmcreid/256_resnet50' \
DATALOADER.USE_RESAMPLING False > train_1p.log 2>&1 &
