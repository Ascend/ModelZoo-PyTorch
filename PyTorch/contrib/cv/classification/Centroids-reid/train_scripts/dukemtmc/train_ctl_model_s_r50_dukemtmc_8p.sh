source env_npu.sh

export RANK_SIZE=8
export WORLD_SIZE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29680

for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
do
    export RANK=$RANK_ID
    export LOCAL_RANK=$RANK

    python3 train_ctl_model.py \
    --config_file="configs/256_resnet50.yml" \
    GPU_IDS [$RANK] \
    DATASETS.NAMES 'DukeMTMC-reID' \
    DATASETS.ROOT_DIR '/home/xyc/data/' \
    SOLVER.IMS_PER_BATCH 16 \
    TEST.IMS_PER_BATCH 128 \
    SOLVER.BASE_LR 0.00035 \
    OUTPUT_DIR './logs/dukemtmcreid/256_resnet50' \
    DATALOADER.USE_RESAMPLING False > train_8p_${RANK}.log 2>&1 &
done
