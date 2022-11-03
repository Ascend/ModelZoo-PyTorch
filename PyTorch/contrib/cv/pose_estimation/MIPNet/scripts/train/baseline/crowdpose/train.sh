cd ../../../..

# # # # # # # # # # # -----------------------------------------------------------

# # ## uncomment for hrnet w48-384
# # # # # # # # # # -----------------------------------------------------------
 CUDA_VISIBLE_DEVICES=0,1,2, python tools/train.py \
     --cfg experiments/crowdpose/hrnet/w48_384x288_adam_lr1e-3.yaml \
     GPUS '(0,1,2)' \
     OUTPUT_DIR 'Outputs/outputs/crowdpose'\
     LOG_DIR 'Outputs/logs/crowdpose'\
     DATASET.DATASET 'crowdpose' \
     DATASET.TRAIN_DATASET 'crowdpose' \
     DATASET.TRAIN_SET 'train' \
     DATASET.TRAIN_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
     DATASET.TRAIN_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_train.json' \
     DATASET.TRAIN_DATASET_TYPE 'crowdpose' \
     DATASET.TEST_DATASET 'crowdpose' \
     DATASET.TEST_SET 'val' \
     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_val.json' \
     DATASET.TEST_DATASET_TYPE 'crowdpose' \
     TRAIN.LR 0.001 \
     TRAIN.END_EPOCH 210 \
     TRAIN.BATCH_SIZE_PER_GPU 10 \
     TEST.BATCH_SIZE_PER_GPU 256 \
     TEST.USE_GT_BBOX True \
     EPOCH_EVAL_FREQ 1 \
     PRINT_FREQ 100 \
     MODEL.NAME 'pose_hrnet' \


# # ## uncomment for hrnet w48-256
# # # # # # # # # # # -----------------------------------------------------------
#  CUDA_VISIBLE_DEVICES=0,1,2, python tools/train.py \
#      --cfg experiments/crowdpose/hrnet/w48_256x192_adam_lr1e-3.yaml \
#      GPUS '(0,1,2)' \
#      OUTPUT_DIR 'Outputs/outputs/crowdpose'\
#      LOG_DIR 'Outputs/logs/crowdpose'\
#      DATASET.DATASET 'crowdpose' \
#      DATASET.TRAIN_DATASET 'crowdpose' \
#      DATASET.TRAIN_SET 'train' \
#      DATASET.TRAIN_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
#      DATASET.TRAIN_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_train.json' \
#      DATASET.TRAIN_DATASET_TYPE 'crowdpose' \
#      DATASET.TEST_DATASET 'crowdpose' \
#      DATASET.TEST_SET 'val' \
#      DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
#      DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_val.json' \
#      DATASET.TEST_DATASET_TYPE 'crowdpose' \
#      TRAIN.LR 0.001 \
#      TRAIN.END_EPOCH 210 \
#      TRAIN.BATCH_SIZE_PER_GPU 10 \
#      TEST.BATCH_SIZE_PER_GPU 256 \
#      TEST.USE_GT_BBOX True \
#      EPOCH_EVAL_FREQ 1 \
#      PRINT_FREQ 100 \
#      MODEL.NAME 'pose_hrnet' \



# # ## uncomment for hrnet w32-384
# # # # # # # # # # # -----------------------------------------------------------
#  CUDA_VISIBLE_DEVICES=0,1,2, python tools/train.py \
#      --cfg experiments/crowdpose/hrnet/w32_384x288_adam_lr1e-3.yaml \
#      GPUS '(0,1,2)' \
#      OUTPUT_DIR 'Outputs/outputs/crowdpose'\
#      LOG_DIR 'Outputs/logs/crowdpose'\
#      DATASET.DATASET 'crowdpose' \
#      DATASET.TRAIN_DATASET 'crowdpose' \
#      DATASET.TRAIN_SET 'train' \
#      DATASET.TRAIN_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
#      DATASET.TRAIN_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_train.json' \
#      DATASET.TRAIN_DATASET_TYPE 'crowdpose' \
#      DATASET.TEST_DATASET 'crowdpose' \
#      DATASET.TEST_SET 'val' \
#      DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
#      DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_val.json' \
#      DATASET.TEST_DATASET_TYPE 'crowdpose' \
#      TRAIN.LR 0.001 \
#      TRAIN.END_EPOCH 210 \
#      TRAIN.BATCH_SIZE_PER_GPU 64 \
#      TEST.BATCH_SIZE_PER_GPU 256 \
#      TEST.USE_GT_BBOX True \
#      EPOCH_EVAL_FREQ 1 \
#      PRINT_FREQ 100 \
#      MODEL.NAME 'pose_hrnet' \


# # ## uncomment for hrnet w32-256
# # # # # # # # # # # -----------------------------------------------------------
#  CUDA_VISIBLE_DEVICES=0,1,2, python tools/train.py \
#      --cfg experiments/crowdpose/hrnet/w32_256x192_adam_lr1e-3.yaml \
#      GPUS '(0,1,2)' \
#      OUTPUT_DIR 'Outputs/outputs/crowdpose'\
#      LOG_DIR 'Outputs/logs/crowdpose'\
#      DATASET.DATASET 'crowdpose' \
#      DATASET.TRAIN_DATASET 'crowdpose' \
#      DATASET.TRAIN_SET 'train' \
#      DATASET.TRAIN_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
#      DATASET.TRAIN_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_train.json' \
#      DATASET.TRAIN_DATASET_TYPE 'crowdpose' \
#      DATASET.TEST_DATASET 'crowdpose' \
#      DATASET.TEST_SET 'val' \
#      DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
#      DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_val.json' \
#      DATASET.TEST_DATASET_TYPE 'crowdpose' \
#      TRAIN.LR 0.001 \
#      TRAIN.END_EPOCH 210 \
#      TRAIN.BATCH_SIZE_PER_GPU 64 \
#      TEST.BATCH_SIZE_PER_GPU 256 \
#      TEST.USE_GT_BBOX True \
#      EPOCH_EVAL_FREQ 1 \
#      PRINT_FREQ 100 \
#      MODEL.NAME 'pose_hrnet' \


