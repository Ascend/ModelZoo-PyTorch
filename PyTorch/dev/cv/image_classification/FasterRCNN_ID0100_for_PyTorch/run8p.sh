source ./env.sh
export PYTHONPATH=./:$PYTHONPATH

nohup python3.7 tools/train_net.py \
        --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml \
        --device-ids 0 1 2 3 4 5 6 7 \
        --num-gpus 8\
        AMP 1\
        OPT_LEVEL O2 \
        LOSS_SCALE_VALUE 64 \
        SOLVER.IMS_PER_BATCH 64 \
        SOLVER.MAX_ITER 10250 \
        SEED 1234 \
        MODEL.RPN.NMS_THRESH 0.8 \
        MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO 2 \
        MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO 2 \
        DATALOADER.NUM_WORKERS 8 \
        SOLVER.BASE_LR 0.02 &
