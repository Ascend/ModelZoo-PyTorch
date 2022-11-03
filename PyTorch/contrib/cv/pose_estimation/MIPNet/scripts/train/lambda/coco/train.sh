cd ../../../..

# # # # # # # # # # -----------------------------------------------------------
#  CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/lambda/train_lambda_real.py \
#      --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
#      GPUS '(0,1,2,3)' \
#      OUTPUT_DIR 'Outputs/outputs/lambda/lambda_coco_real'\
#      LOG_DIR 'Outputs/logs/lambda/lambda_coco_real'\
#      TEST.MODEL_FILE 'models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth' \
#      DATASET.TRAIN_DATASET 'coco_lambda' \
#      DATASET.TRAIN_SET 'train2017' \
#      DATASET.TRAIN_IMAGE_DIR '/home/ANT.AMAZON.COM/khiXYZ/Desktop/datasets/coco/images/train2017'\
#      DATASET.TRAIN_ANNOTATION_FILE '/home/ANT.AMAZON.COM/khiXYZ/Desktop/datasets/coco/annotations/person_keypoints_train2017.json' \
#      DATASET.TRAIN_DATASET_TYPE 'coco_lambda' \
#      DATASET.TEST_DATASET 'coco' \
#      DATASET.TEST_SET 'val2017' \
#      DATASET.TEST_IMAGE_DIR '/home/ANT.AMAZON.COM/khiXYZ/Desktop/datasets/coco/images/val2017'\
#      DATASET.TEST_ANNOTATION_FILE '/home/ANT.AMAZON.COM/khiXYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
#      DATASET.TEST_DATASET_TYPE 'coco' \
#      TRAIN.LR 0.001 \
#      TRAIN.END_EPOCH 210 \
#      TRAIN.BATCH_SIZE_PER_GPU 32 \
#      TEST.BATCH_SIZE_PER_GPU 256 \
#      TEST.USE_GT_BBOX True \
#      EPOCH_EVAL_FREQ 1 \
#      PRINT_FREQ 100 \
#      MODEL.NAME 'pose_hrnet_se_lambda' \
#      MODEL.SE_MODULES '[True, True, True, True]'

# # # # # # # # # -----------------------------------------------------------
## train batch size 32 does not fit in for w48
## train batch size 10 maybe?
### test batch size: 64

 CUDA_VISIBLE_DEVICES=0,1,2,3, python tools/lambda/train_lambda_real.py \
     --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml \
     GPUS '(0,1,2,3,)' \
     OUTPUT_DIR 'Outputs/outputs/lambda/lambda_coco_real_waffle'\
     LOG_DIR 'Outputs/logs/lambda/lambda_coco_real_waffle'\
     TEST.MODEL_FILE 'models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth' \
     DATASET.TRAIN_DATASET 'coco_lambda' \
     DATASET.TRAIN_SET 'train2017' \
     DATASET.TRAIN_IMAGE_DIR '/mnt/nas/rawal/Desktop/MIPNet/coco/images/train2017'\
     DATASET.TRAIN_ANNOTATION_FILE '/mnt/nas/rawal/Desktop/MIPNet/coco/annotations/person_keypoints_train2017.json' \
     DATASET.TRAIN_DATASET_TYPE 'coco_lambda' \
     DATASET.TEST_DATASET 'coco' \
     DATASET.TEST_SET 'val2017' \
     DATASET.TEST_IMAGE_DIR '/mnt/nas/rawal/Desktop/MIPNet/coco/images/val2017'\
     DATASET.TEST_ANNOTATION_FILE '/mnt/nas/rawal/Desktop/MIPNet/coco/annotations/person_keypoints_val2017.json' \
     DATASET.TEST_DATASET_TYPE 'coco' \
     TRAIN.LR 0.001 \
     TRAIN.BEGIN_EPOCH 0 \
     TRAIN.END_EPOCH 110 \
     TRAIN.LR_STEP '(70, 100)' \
     TRAIN.BATCH_SIZE_PER_GPU 10 \
     TEST.BATCH_SIZE_PER_GPU 64 \
     TEST.USE_GT_BBOX True \
     EPOCH_EVAL_FREQ 1 \
     PRINT_FREQ 100 \
     MODEL.NAME 'pose_hrnet_se_lambda' \
     MODEL.SE_MODULES '[False, False, True, True]'


