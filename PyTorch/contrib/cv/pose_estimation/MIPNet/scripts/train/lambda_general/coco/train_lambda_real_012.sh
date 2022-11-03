cd ../../../..

# # # # # # # # # # # # -----------------------------------------------------------
#  CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda_general/train_lambda_real_012.py \
#      --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml \
#      GPUS '(0,1,2)' \
#      OUTPUT_DIR 'Outputs/outputs/lambda_general/lambda_coco_real'\
#      LOG_DIR 'Outputs/logs/lambda_general/lambda_coco_real'\
#      TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth' \
#      DATASET.TRAIN_DATASET 'coco_lambda_012' \
#      DATASET.TRAIN_SET 'train2017' \
#      DATASET.TRAIN_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/train2017'\
#      DATASET.TRAIN_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_train2017.json' \
#      DATASET.TRAIN_DATASET_TYPE 'coco_lambda_012' \
#      DATASET.TEST_DATASET 'coco' \
#      DATASET.TEST_SET 'val2017' \
#      DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/val2017'\
#      DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
#      DATASET.TEST_DATASET_TYPE 'coco' \
#      TRAIN.LR 0.001 \
#      TRAIN.BEGIN_EPOCH 0 \
#      TRAIN.END_EPOCH 110 \
#      TRAIN.LR_STEP '(70, 100)' \
#      TRAIN.BATCH_SIZE_PER_GPU 16 \
#      TEST.BATCH_SIZE_PER_GPU 256 \
#      TEST.USE_GT_BBOX True \
#      EPOCH_EVAL_FREQ 1 \
#      PRINT_FREQ 100 \
#      MODEL.NAME 'pose_hrnet_se_lambda' \
#      MODEL.SE_MODULES '[False, False, True, True]'


# # # # # # # # # # # -----------------------------------------------------------
# # # # # # # # # # # -----------------------------------------------------------
 CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda_general/train_lambda_real_012.py \
     --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
     GPUS '(0,1,2)' \
     OUTPUT_DIR 'Outputs/outputs/lambda_general/lambda_coco_real_01'\
     LOG_DIR 'Outputs/logs/lambda_general/lambda_coco_real_01'\
     TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth' \
     DATASET.TRAIN_DATASET 'coco_lambda_012' \
     DATASET.TRAIN_SET 'train2017' \
     DATASET.TRAIN_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/train2017'\
     DATASET.TRAIN_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_train2017.json' \
     DATASET.TRAIN_DATASET_TYPE 'coco_lambda_012' \
     DATASET.TEST_DATASET 'coco' \
     DATASET.TEST_SET 'val2017' \
     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/val2017'\
     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
     DATASET.TEST_DATASET_TYPE 'coco' \
     TRAIN.LR 0.001 \
     TRAIN.BEGIN_EPOCH 0 \
     TRAIN.END_EPOCH 110 \
     TRAIN.LR_STEP '(70, 100)' \
     TRAIN.BATCH_SIZE_PER_GPU 20 \
     TEST.BATCH_SIZE_PER_GPU 256 \
     TEST.USE_GT_BBOX True \
     EPOCH_EVAL_FREQ 1 \
     PRINT_FREQ 100 \
     MODEL.NAME 'pose_hrnet_se_lambda' \
     MODEL.SE_MODULES '[False, False, True, True]'





