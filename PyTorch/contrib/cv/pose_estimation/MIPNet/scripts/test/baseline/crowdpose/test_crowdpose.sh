cd ../../../..

# # # -----------------------------------------------------------
# # # ---------------------------------------
# # # # ---------------------------------------
#  CUDA_VISIBLE_DEVICES=0,1,2 python tools/test.py \
#      --cfg experiments/crowdpose/hrnet/w32_256x192_adam_lr1e-3.yaml \
#      GPUS '(0,1,2)' \
#      TEST.MODEL_FILE 'models/lambda/crowdpose/Logs/w32/256/baseline/checkpoint_106.pth' \
#      OUTPUT_DIR 'Outputs/outputs/crowdpose/hrnet' \
#      LOG_DIR 'Outputs/logs/crowdpose/hrnet' \
#      DATASET.DATASET 'crowdpose' \
#      DATASET.TEST_DATASET 'crowdpose' \
#      DATASET.TEST_SET 'val' \
#      DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
#      DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_val.json' \
#      DATASET.TEST_DATASET_TYPE 'crowdpose' \
#      TEST.USE_GT_BBOX True \
#      TEST.BATCH_SIZE_PER_GPU 256 \
#      TEST.POST_PROCESS True \

# # # ---------------------------------------
 CUDA_VISIBLE_DEVICES=0,1,2 python tools/test.py \
     --cfg experiments/crowdpose/hrnet/w48_256x192_adam_lr1e-3.yaml \
     GPUS '(0,1,2)' \
     TEST.MODEL_FILE 'models/lambda/crowdpose/Logs/w48/256/baseline/checkpoint_107.pth' \
     OUTPUT_DIR 'Outputs/outputs/crowdpose/hrnet' \
     LOG_DIR 'Outputs/logs/crowdpose/hrnet' \
     DATASET.DATASET 'crowdpose' \
     DATASET.TEST_DATASET 'crowdpose' \
     DATASET.TEST_SET 'val' \
     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_val.json' \
     DATASET.TEST_DATASET_TYPE 'crowdpose' \
     TEST.USE_GT_BBOX True \
     TEST.BATCH_SIZE_PER_GPU 256 \
     TEST.POST_PROCESS True \

# # # # ---------------------------------------
#  CUDA_VISIBLE_DEVICES=0,1,2 python tools/test.py \
#      --cfg experiments/crowdpose/hrnet/w32_384x288_adam_lr1e-3.yaml \
#      GPUS '(0,1,2)' \
#      TEST.MODEL_FILE 'models/lambda/crowdpose/Logs/w32/384/baseline/checkpoint_108.pth' \
#      OUTPUT_DIR 'Outputs/outputs/crowdpose/hrnet' \
#      LOG_DIR 'Outputs/logs/crowdpose/hrnet' \
#      DATASET.DATASET 'crowdpose' \
#      DATASET.TEST_DATASET 'crowdpose' \
#      DATASET.TEST_SET 'val' \
#      DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
#      DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_val.json' \
#      DATASET.TEST_DATASET_TYPE 'crowdpose' \
#      TEST.USE_GT_BBOX True \
#      TEST.BATCH_SIZE_PER_GPU 256 \
#      TEST.POST_PROCESS True \


# # # # ---------------------------------------
#  CUDA_VISIBLE_DEVICES=0,1,2 python tools/test.py \
#      --cfg experiments/crowdpose/hrnet/w48_384x288_adam_lr1e-3.yaml \
#      GPUS '(0,1,2)' \
#      TEST.MODEL_FILE 'models/lambda/crowdpose/Logs/w48/384/baseline/checkpoint_108.pth' \
#      OUTPUT_DIR 'Outputs/outputs/crowdpose/hrnet' \
#      LOG_DIR 'Outputs/logs/crowdpose/hrnet' \
#      DATASET.DATASET 'crowdpose' \
#      DATASET.TEST_DATASET 'crowdpose' \
#      DATASET.TEST_SET 'val' \
#      DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
#      DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_val.json' \
#      DATASET.TEST_DATASET_TYPE 'crowdpose' \
#      TEST.USE_GT_BBOX True \
#      TEST.BATCH_SIZE_PER_GPU 256 \
#      TEST.POST_PROCESS True \


# # # ---------------------------------------
 # CUDA_VISIBLE_DEVICES=0,1,2 python tools/test.py \
 #     --cfg experiments/crowdpose/hrnet/w48_384x288_adam_lr1e-3.yaml \
 #     GPUS '(0,1,2)' \
 #     TEST.MODEL_FILE 'models/lambda/crowdpose/Logs/w48/384/baseline/checkpoint_108.pth' \
 #     OUTPUT_DIR 'Outputs/outputs/crowdpose/hrnet' \
 #     LOG_DIR 'Outputs/logs/crowdpose/hrnet' \
 #     DATASET.DATASET 'crowdpose' \
 #     DATASET.TEST_DATASET 'crowdpose' \
 #     DATASET.TEST_SET 'val' \
 #     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/crowdpose/images'\
 #     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/crowdpose/json/crowdpose_test.json' \
 #     DATASET.TEST_DATASET_TYPE 'crowdpose' \
 #     TEST.USE_GT_BBOX True \
 #     TEST.BATCH_SIZE_PER_GPU 256 \
 #     TEST.POST_PROCESS True \


