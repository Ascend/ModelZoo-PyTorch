cd ../../../..

# # # -----------------------------------------------------------
# # # # ---------------------------------------
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/test.py \
#      --cfg experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml \
#      GPUS '(0,1,2)' \
#      OUTPUT_DIR 'Outputs/outputs/ochuman/hrnet/' \
#      LOG_DIR 'Outputs/logs/ochuman/hrnet' \
#      TEST.MODEL_FILE 'models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth' \
#      DATASET.TEST_DATASET 'ochuman' \
#      DATASET.TEST_SET 'val2017' \
#      DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/OCHuman/images'\
#      DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json' \
#      DATASET.TEST_DATASET_TYPE 'ochuman' \
#      TEST.USE_GT_BBOX True \
#      TEST.BATCH_SIZE_PER_GPU 256 \
#      TEST.POST_PROCESS True \

# CUDA_VISIBLE_DEVICES=0,1,2 python tools/test.py \
#      --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
#      GPUS '(0,1,2)' \
#      OUTPUT_DIR 'Outputs/outputs/ochuman/hrnet/' \
#      LOG_DIR 'Outputs/logs/ochuman/hrnet' \
#      TEST.MODEL_FILE 'models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth' \
#      DATASET.TEST_DATASET 'ochuman' \
#      DATASET.TEST_SET 'val2017' \
#      DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/OCHuman/images'\
#      DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json' \
#      DATASET.TEST_DATASET_TYPE 'ochuman' \
#      TEST.USE_GT_BBOX True \
#      TEST.BATCH_SIZE_PER_GPU 256 \
#      TEST.POST_PROCESS True \


# CUDA_VISIBLE_DEVICES=0,1,2 python tools/test.py \
#      --cfg experiments/coco/resnet/res152_256x192_d256x3_adam_lr1e-3.yaml \
#      GPUS '(0,1,2)' \
#      OUTPUT_DIR 'Outputs/outputs/ochuman/hrnet/' \
#      LOG_DIR 'Outputs/logs/ochuman/hrnet' \
#      TEST.MODEL_FILE 'models/pytorch/pose_coco/pose_resnet_152_256x192.pth' \
#      DATASET.TEST_DATASET 'ochuman' \
#      DATASET.TEST_SET 'val2017' \
#      DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/OCHuman/images'\
#      DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json' \
#      DATASET.TEST_DATASET_TYPE 'ochuman' \
#      TEST.USE_GT_BBOX True \
#      TEST.BATCH_SIZE_PER_GPU 256 \
#      TEST.POST_PROCESS True \


CUDA_VISIBLE_DEVICES=0,1,2 python tools/test.py \
     --cfg experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml \
     GPUS '(0,1,2)' \
     OUTPUT_DIR 'Outputs/supp/baseline/' \
     LOG_DIR 'Outputs/supp/baseline' \
     TEST.MODEL_FILE 'models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth' \
     DATASET.TEST_DATASET 'ochuman' \
     DATASET.TEST_SET 'val2017' \
     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/OCHuman/images'\
     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json' \
     DATASET.TEST_DATASET_TYPE 'ochuman' \
     TEST.USE_GT_BBOX True \
     TEST.BATCH_SIZE_PER_GPU 256 \
     TEST.POST_PROCESS True \
     TEST.OKS_THRE 1.0 \