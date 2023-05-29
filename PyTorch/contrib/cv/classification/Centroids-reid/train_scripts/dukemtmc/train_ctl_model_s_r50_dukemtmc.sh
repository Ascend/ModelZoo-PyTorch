source env_npu.sh

python3 train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0,1] \
DATASETS.NAMES 'DukeMTMC-reID' \
DATASETS.ROOT_DIR '/home/xyc/data/' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs/dukemtmcreid/256_resnet50' \
DATALOADER.USE_RESAMPLING False \
