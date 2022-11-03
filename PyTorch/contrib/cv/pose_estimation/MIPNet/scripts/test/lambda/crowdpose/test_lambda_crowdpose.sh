cd ../../../..

# # # # -----------------------------------------------------------
# # # -----------------------------------------------------------
# # # # # # # # # # # -----------------------------------------------------------
#  CUDA_VISIBLE_DEVICES=0,1,2, python tools/lambda/test_lambda.py \
#      --cfg experiments/crowdpose/hrnet/w32_256x192_adam_lr1e-3.yaml \
#      GPUS '(0,1,2,)' \
#      TEST.MODEL_FILE 'models/lambda/crowdpose/Logs/w32/256/checkpoint_102.pth' \
#      OUTPUT_DIR 'Outputs/outputs/crowdpose/hrnet' \
#      LOG_DIR 'Outputs/logs/crowdpose/hrnet' \
#      DATASET.DATASET 'crowdpose' \
#      DATASET.TEST_DATASET 'crowdpose' \
#      DATASET.TEST_SET 'val' \
#      DATASET.TEST_IMAGE_DIR '/home/rawal/Desktop/datasets/crowdpose/images'\
#      DATASET.TEST_ANNOTATION_FILE '/home/rawal/Desktop/datasets/crowdpose/json/crowdpose_val.json' \
#      DATASET.TEST_DATASET_TYPE 'crowdpose' \
#      TEST.BATCH_SIZE_PER_GPU 256 \
#      TEST.USE_GT_BBOX True \
#      TEST.POST_PROCESS True \
#      DATASET.TEST_DATASET_TYPE 'crowdpose' \
#      PRINT_FREQ 100 \
#      MODEL.NAME 'pose_hrnet_se_lambda' \
#      MODEL.SE_MODULES '[False, False, True, True]'

# # # # # # # # # # # -----------------------------------------------------------
#  CUDA_VISIBLE_DEVICES=0,1,2, python tools/lambda/test_lambda.py \
#      --cfg experiments/crowdpose/hrnet/w48_256x192_adam_lr1e-3.yaml \
#      GPUS '(0,1,2,)' \
#      TEST.MODEL_FILE 'models/lambda/crowdpose/Logs/w48/256/checkpoint_103.pth' \
#      OUTPUT_DIR 'Outputs/outputs/crowdpose/hrnet' \
#      LOG_DIR 'Outputs/logs/crowdpose/hrnet' \
#      DATASET.DATASET 'crowdpose' \
#      DATASET.TEST_DATASET 'crowdpose' \
#      DATASET.TEST_SET 'val' \
#      DATASET.TEST_IMAGE_DIR '/home/rawal/Desktop/datasets/crowdpose/images'\
#      DATASET.TEST_ANNOTATION_FILE '/home/rawal/Desktop/datasets/crowdpose/json/crowdpose_val.json' \
#      DATASET.TEST_DATASET_TYPE 'crowdpose' \
#      TEST.BATCH_SIZE_PER_GPU 256 \
#      TEST.USE_GT_BBOX True \
#      TEST.POST_PROCESS True \
#      DATASET.TEST_DATASET_TYPE 'crowdpose' \
#      PRINT_FREQ 100 \
#      MODEL.NAME 'pose_hrnet_se_lambda' \
#      MODEL.SE_MODULES '[False, False, True, True]'

# # # # # # # # # # # -----------------------------------------------------------
#  CUDA_VISIBLE_DEVICES=0,1,2, python tools/lambda/test_lambda.py \
#      --cfg experiments/crowdpose/hrnet/w32_384x288_adam_lr1e-3.yaml \
#      GPUS '(0,1,2,)' \
#      TEST.MODEL_FILE 'models/lambda/crowdpose/Logs/w32/384/checkpoint_102.pth' \
#      OUTPUT_DIR 'Outputs/outputs/crowdpose/hrnet' \
#      LOG_DIR 'Outputs/logs/crowdpose/hrnet' \
#      DATASET.DATASET 'crowdpose' \
#      DATASET.TEST_DATASET 'crowdpose' \
#      DATASET.TEST_SET 'val' \
#      DATASET.TEST_IMAGE_DIR '/home/rawal/Desktop/datasets/crowdpose/images'\
#      DATASET.TEST_ANNOTATION_FILE '/home/rawal/Desktop/datasets/crowdpose/json/crowdpose_val.json' \
#      DATASET.TEST_DATASET_TYPE 'crowdpose' \
#      TEST.BATCH_SIZE_PER_GPU 256 \
#      TEST.USE_GT_BBOX True \
#      TEST.POST_PROCESS True \
#      DATASET.TEST_DATASET_TYPE 'crowdpose' \
#      PRINT_FREQ 100 \
#      MODEL.NAME 'pose_hrnet_se_lambda' \
#      MODEL.SE_MODULES '[False, False, True, True]'


# # # # # # # # # # -----------------------------------------------------------
 CUDA_VISIBLE_DEVICES=0,1,2, python tools/lambda/test_lambda.py \
     --cfg experiments/crowdpose/hrnet/w48_384x288_adam_lr1e-3.yaml \
     GPUS '(0,1,2,)' \
     TEST.MODEL_FILE 'models/lambda/crowdpose/Logs/w48/384/checkpoint_109.pth' \
     OUTPUT_DIR 'Outputs/outputs/crowdpose/hrnet' \
     LOG_DIR 'Outputs/logs/crowdpose/hrnet' \
     DATASET.DATASET 'crowdpose' \
     DATASET.TEST_DATASET 'crowdpose' \
     DATASET.TEST_SET 'val' \
     DATASET.TEST_IMAGE_DIR '/home/rawal/Desktop/datasets/crowdpose/images'\
     DATASET.TEST_ANNOTATION_FILE '/home/rawal/Desktop/datasets/crowdpose/json/crowdpose_val.json' \
     DATASET.TEST_DATASET_TYPE 'crowdpose' \
     TEST.BATCH_SIZE_PER_GPU 256 \
     TEST.USE_GT_BBOX True \
     TEST.POST_PROCESS True \
     PRINT_FREQ 100 \
     MODEL.NAME 'pose_hrnet_se_lambda' \
     MODEL.SE_MODULES '[False, False, True, True]'

# # # -----------------------------------------------------------
# # # -----------------------------------------------------------










