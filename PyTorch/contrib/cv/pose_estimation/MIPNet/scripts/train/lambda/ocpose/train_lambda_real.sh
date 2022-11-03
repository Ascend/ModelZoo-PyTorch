cd ../../../..


# # # # # # # # # -----------------------------------------------------------
 CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/train_lambda_real.py \
     --cfg experiments/ocpose/hrnet/w32_256x192_adam_lr1e-3.yaml \
     GPUS '(0,1,2)' \
     OUTPUT_DIR 'Outputs/outputs/lambda/ocpose'\
     LOG_DIR 'Outputs/logs/lambda/ocpose'\
     DATASET.TRAIN_DATASET 'ocpose_lambda' \
     DATASET.TRAIN_SET 'train2017' \
     DATASET.TRAIN_IMAGE_DIR '/home/XYZ/Desktop/datasets/OCPose/images'\
     DATASET.TRAIN_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/OCPose/annotations/annot_train.json' \
     DATASET.TRAIN_DATASET_TYPE 'ocpose_lambda' \
     DATASET.TEST_DATASET 'ocpose' \
     DATASET.TEST_SET 'val2017' \
     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/OCPose/images'\
     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/annot_val.json' \
     DATASET.TEST_DATASET_TYPE 'coco' \
     TRAIN.LR 0.001 \
     TRAIN.BEGIN_EPOCH 0 \
     TRAIN.END_EPOCH 210 \
     TRAIN.BATCH_SIZE_PER_GPU 32 \
     TEST.BATCH_SIZE_PER_GPU 256 \
     TEST.USE_GT_BBOX True \
     EPOCH_EVAL_FREQ 1 \
     PRINT_FREQ 100 \
     MODEL.NAME 'pose_hrnet_se_lambda' \
     MODEL.SE_MODULES '[False, False, True, True]'


