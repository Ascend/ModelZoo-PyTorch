cd ../../../..

# ## uncomment for hrnet w48-384
# # # # # # # # # # -----------------------------------------------------------
#  CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/train_lambda_real.py \
#      --cfg experiments/crowdpose/hrnet/w48_384x288_adam_lr1e-3.yaml \
#      GPUS '(0,1,2)' \
#      OUTPUT_DIR 'Outputs/outputs/lambda/lambda_crowdpose_real'\
#      LOG_DIR 'Outputs/logs/lambda/lambda_crowdpose_real'\
#      DATASET.DATASET 'crowdpose' \
#      DATASET.TRAIN_DATASET 'crowdpose_lambda' \
#      DATASET.TRAIN_SET 'train' \
#      DATASET.TRAIN_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
#      DATASET.TRAIN_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_train.json' \
#      DATASET.TRAIN_DATASET_TYPE 'crowdpose_lambda' \
#      DATASET.TEST_DATASET 'crowdpose' \
#      DATASET.TEST_SET 'val' \
#      DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
#      DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_val.json' \
#      DATASET.TEST_DATASET_TYPE 'crowdpose' \
#      TRAIN.LR 0.001 \
#      TRAIN.BEGIN_EPOCH 0 \
#      TRAIN.END_EPOCH 210 \
#      TRAIN.BATCH_SIZE_PER_GPU 10 \
#      TEST.BATCH_SIZE_PER_GPU 256 \
#      TEST.USE_GT_BBOX True \
#      EPOCH_EVAL_FREQ 1 \
#      PRINT_FREQ 100 \
#      MODEL.NAME 'pose_hrnet_se_lambda' \
#      MODEL.SE_MODULES '[False, False, True, True]'


# ## uncomment for hrnet w48-256
# # # # # # # # # # -----------------------------------------------------------
#  CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/train_lambda_real.py \
#      --cfg experiments/crowdpose/hrnet/w48_256x192_adam_lr1e-3.yaml \
#      GPUS '(0,1,2)' \
#      OUTPUT_DIR 'Outputs/outputs/lambda/lambda_crowdpose_real'\
#      LOG_DIR 'Outputs/logs/lambda/lambda_crowdpose_real'\
#      DATASET.DATASET 'crowdpose' \
#      DATASET.TRAIN_DATASET 'crowdpose_lambda' \
#      DATASET.TRAIN_SET 'train' \
#      DATASET.TRAIN_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
#      DATASET.TRAIN_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_train.json' \
#      DATASET.TRAIN_DATASET_TYPE 'crowdpose_lambda' \
#      DATASET.TEST_DATASET 'crowdpose' \
#      DATASET.TEST_SET 'val' \
#      DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
#      DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_val.json' \
#      DATASET.TEST_DATASET_TYPE 'crowdpose' \
#      TRAIN.LR 0.001 \
#      TRAIN.BEGIN_EPOCH 0 \
#      TRAIN.END_EPOCH 210 \
#      TRAIN.BATCH_SIZE_PER_GPU 10 \
#      TEST.BATCH_SIZE_PER_GPU 256 \
#      TEST.USE_GT_BBOX True \
#      EPOCH_EVAL_FREQ 1 \
#      PRINT_FREQ 100 \
#      MODEL.NAME 'pose_hrnet_se_lambda' \
#      MODEL.SE_MODULES '[False, False, True, True]'

# ## uncomment for hrnet w32-384
# # # # # # # # # # -----------------------------------------------------------
#  CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/train_lambda_real.py \
#      --cfg experiments/crowdpose/hrnet/w32_384x288_adam_lr1e-3.yaml \
#      GPUS '(0,1,2)' \
#      OUTPUT_DIR 'Outputs/outputs/lambda/lambda_crowdpose_real'\
#      LOG_DIR 'Outputs/logs/lambda/lambda_crowdpose_real'\
#      DATASET.DATASET 'crowdpose' \
#      DATASET.TRAIN_DATASET 'crowdpose_lambda' \
#      DATASET.TRAIN_SET 'train' \
#      DATASET.TRAIN_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
#      DATASET.TRAIN_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_train.json' \
#      DATASET.TRAIN_DATASET_TYPE 'crowdpose_lambda' \
#      DATASET.TEST_DATASET 'crowdpose' \
#      DATASET.TEST_SET 'val' \
#      DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
#      DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_val.json' \
#      DATASET.TEST_DATASET_TYPE 'crowdpose' \
#      TRAIN.LR 0.001 \
#      TRAIN.BEGIN_EPOCH 0 \
#      TRAIN.END_EPOCH 210 \
#      TRAIN.BATCH_SIZE_PER_GPU 32 \
#      TEST.BATCH_SIZE_PER_GPU 256 \
#      TEST.USE_GT_BBOX True \
#      EPOCH_EVAL_FREQ 1 \
#      PRINT_FREQ 100 \
#      MODEL.NAME 'pose_hrnet_se_lambda' \
#      MODEL.SE_MODULES '[False, False, True, True]'


# ## uncomment for hrnet w32-256
# # # # # # # # # # -----------------------------------------------------------
#  CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/train_lambda_real.py \
#      --cfg experiments/crowdpose/hrnet/w32_256x192_adam_lr1e-3.yaml \
#      GPUS '(0,1,2)' \
#      OUTPUT_DIR 'Outputs/outputs/lambda/lambda_crowdpose_real'\
#      LOG_DIR 'Outputs/logs/lambda/lambda_crowdpose_real'\
#      DATASET.DATASET 'crowdpose' \
#      DATASET.TRAIN_DATASET 'crowdpose_lambda' \
#      DATASET.TRAIN_SET 'train' \
#      DATASET.TRAIN_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
#      DATASET.TRAIN_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_train.json' \
#      DATASET.TRAIN_DATASET_TYPE 'crowdpose_lambda' \
#      DATASET.TEST_DATASET 'crowdpose' \
#      DATASET.TEST_SET 'val' \
#      DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
#      DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_val.json' \
#      DATASET.TEST_DATASET_TYPE 'crowdpose' \
#      TRAIN.LR 0.001 \
#      TRAIN.BEGIN_EPOCH 0 \
#      TRAIN.END_EPOCH 210 \
#      TRAIN.BATCH_SIZE_PER_GPU 32 \
#      TEST.BATCH_SIZE_PER_GPU 256 \
#      TEST.USE_GT_BBOX True \
#      EPOCH_EVAL_FREQ 1 \
#      PRINT_FREQ 100 \
#      MODEL.NAME 'pose_hrnet_se_lambda' \
#      MODEL.SE_MODULES '[False, False, True, True]'



## uncomment for resnet 101
# # # # # # # # # -----------------------------------------------------------
 CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/train_lambda_real.py \
     --cfg experiments/crowdpose/resnet/res101_384x288_d256x3_adam_lr1e-3.yaml \
     GPUS '(0,1,2)' \
     OUTPUT_DIR 'Outputs/outputs/lambda/resnet101'\
     LOG_DIR 'Outputs/logs/lambda/resnet101'\
     DATASET.DATASET 'crowdpose' \
     DATASET.TRAIN_DATASET 'crowdpose_lambda' \
     DATASET.TRAIN_SET 'train' \
     DATASET.TRAIN_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
     DATASET.TRAIN_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_train.json' \
     DATASET.TRAIN_DATASET_TYPE 'crowdpose_lambda' \
     DATASET.TEST_DATASET 'crowdpose' \
     DATASET.TEST_SET 'val' \
     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_val.json' \
     DATASET.TEST_DATASET_TYPE 'crowdpose' \
     TRAIN.LR 0.001 \
     TRAIN.BEGIN_EPOCH 0 \
     TRAIN.END_EPOCH 210 \
     TRAIN.BATCH_SIZE_PER_GPU 16 \
     TEST.BATCH_SIZE_PER_GPU 256 \
     TEST.USE_GT_BBOX True \
     EPOCH_EVAL_FREQ 1 \
     PRINT_FREQ 100 \
     MODEL.NAME 'pose_resnet_se_lambda' \
     MODEL.SE_MODULES '[False, False, False, True]'