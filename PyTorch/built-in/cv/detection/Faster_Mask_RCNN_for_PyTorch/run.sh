source ./env.sh
export PYTHONPATH=./:$PYTHONPATH

nohup python3.7 tools/train_net.py \
        --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml \
        AMP 1\
        OPT_LEVEL O2 \
        LOSS_SCALE_VALUE 64 \
        MODEL.DEVICE npu:5 \
        SOLVER.IMS_PER_BATCH 8 \
        SOLVER.MAX_ITER 82000 \
        SEED 1234 \
        MODEL.RPN.NMS_THRESH 0.8 \
        MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO 2 \
        MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO 2 \
        DATALOADER.NUM_WORKERS 4 \
        SOLVER.BASE_LR 0.0025 &
