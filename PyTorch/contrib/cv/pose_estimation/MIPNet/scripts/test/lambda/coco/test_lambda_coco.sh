cd ../../../..

# # # # # -----------------------------------------------------------
# # # # # -----------------------------------------------------------
# # # # # # # -----------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/test_lambda.py \
#     --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml \
#     GPUS '(0,1,2)' \
#     OUTPUT_DIR 'Outputs/outputs/lambda/lambda_real_dark/' \
#     LOG_DIR 'Outputs/logs/lambda/lambda_real_dark/' \
#     TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/lambda/w48_384x288/checkpoint_103.pth' \
#     DATASET.TEST_DATASET 'coco' \
#     DATASET.TEST_SET 'val2017' \
#     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/val2017'\
#     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
#     TEST.USE_GT_BBOX True \
#     TEST.BATCH_SIZE_PER_GPU 256 \
#     TEST.POST_PROCESS True \
#     MODEL.NAME 'pose_hrnet_se_lambda' \
#     MODEL.SE_MODULES '[False, False, True, True]' \
#     PRINT_FREQ 100 \
#     TEST.COCO_BBOX_FILE '/home/ANT.AMAZON.COM/khiXYZ/Desktop/datasets/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json'

# CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/test_lambda.py \
#     --cfg experiments/coco/hrnet/w48_256x192_adam_lr1e-3.yaml \
#     GPUS '(0,1,2)' \
#     OUTPUT_DIR 'Outputs/outputs/lambda/lambda_real/' \
#     LOG_DIR 'Outputs/logs/lambda/lambda_real/' \
#     TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/lambda/hrnetw48-256x192/checkpoint_84.pth' \
#     DATASET.TEST_DATASET 'coco' \
#     DATASET.TEST_SET 'val2017' \
#     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/val2017'\
#     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
#     TEST.USE_GT_BBOX True \
#     TEST.BATCH_SIZE_PER_GPU 256 \
#     TEST.POST_PROCESS True \
#     MODEL.NAME 'pose_hrnet_se_lambda' \
#     MODEL.SE_MODULES '[False, False, True, True]' \
#     PRINT_FREQ 100 \
#     TEST.COCO_BBOX_FILE '/home/ANT.AMAZON.COM/khiXYZ/Desktop/datasets/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json'


# CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/test_lambda.py \
#     --cfg experiments/coco/resnet/res152_384x288_d256x3_adam_lr1e-3.yaml \
#     GPUS '(0,1,2)' \
#     OUTPUT_DIR 'Outputs/outputs/lambda/lambda_real/' \
#     LOG_DIR 'Outputs/logs/lambda/lambda_real/' \
#     TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/lambda/resnet152-384x288/checkpoint_96.pth' \
#     DATASET.TEST_DATASET 'coco' \
#     DATASET.TEST_SET 'val2017' \
#     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/val2017'\
#     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
#     TEST.USE_GT_BBOX True \
#     TEST.BATCH_SIZE_PER_GPU 128 \
#     TEST.POST_PROCESS True \
#     MODEL.NAME 'pose_resnet_se_lambda' \
#     MODEL.SE_MODULES '[False, False, True, True]' \
#     PRINT_FREQ 100 \

# CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/test_lambda.py \
#     --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml \
#     GPUS '(0,1,2)' \
#     OUTPUT_DIR 'Outputs/outputs/lambda/lambda_real/' \
#     LOG_DIR 'Outputs/logs/lambda/lambda_real/' \
#     TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/lambda/hrnetw48-384x288/checkpoint_103.pth' \
#     DATASET.TEST_DATASET 'coco' \
#     DATASET.TEST_SET 'test2017' \
#     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/test2017'\
#     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
#     TEST.USE_GT_BBOX True \
#     TEST.BATCH_SIZE_PER_GPU 256 \
#     TEST.POST_PROCESS True \
#     MODEL.NAME 'pose_hrnet_se_lambda' \
#     MODEL.SE_MODULES '[False, False, True, True]' \
#     PRINT_FREQ 100 \
#     TEST.COCO_BBOX_FILE '/home/XYZ/Desktop/datasets/coco/person_detection_results/COCO_test-dev2017_detections_AP_H_609_person.json'


DEVICES=0,1,2,3,
GPUS='(0,1,2,3)'
CONFIG_FILE='experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml'
CHECKPOINT='/mnt/nas/rawal/Desktop/MIPNet/checkpoints/h48_384x288/checkpoint_103.pth'
OUTPUT_DIR='Outputs/outputs/lambda/lambda_real/'
LOG_DIR='Outputs/logs/lambda/lambda_real/'
IMAGE_DIR='/mnt/nas/rawal/Desktop/MIPNet/coco/images/val2017'
ANNOTATION_FILE='/mnt/nas/rawal/Desktop/MIPNet/coco/annotations/person_keypoints_val2017.json'

CUDA_VISIBLE_DEVICES=${DEVICES}  python tools/lambda/test_lambda.py \
    --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml \
    GPUS ${GPUS}  \
    OUTPUT_DIR ${OUTPUT_DIR} \
    LOG_DIR ${LOG_DIR} \
    TEST.MODEL_FILE ${CHECKPOINT} \
    DATASET.TEST_DATASET 'coco' \
    DATASET.TEST_SET 'val2017' \
    DATASET.TEST_IMAGE_DIR ${IMAGE_DIR} \
    DATASET.TEST_ANNOTATION_FILE ${ANNOTATION_FILE} \
    TEST.USE_GT_BBOX True \
    TEST.BATCH_SIZE_PER_GPU 128 \
    TEST.POST_PROCESS True \
    MODEL.NAME 'pose_hrnet_se_lambda' \
    MODEL.SE_MODULES '[False, False, True, True]' \
    PRINT_FREQ 100 \
    TEST.COCO_BBOX_FILE '/home/XYZ/Desktop/datasets/coco/person_detection_results/COCO_test-dev2017_detections_AP_H_609_person.json'



# CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/test_lambda.py \
#     --cfg experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml \
#     GPUS '(0,1,2)' \
#     OUTPUT_DIR 'Outputs/outputs/lambda/lambda_real/' \
#     LOG_DIR 'Outputs/logs/lambda/lambda_real/' \
#     TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/lambda/hrnetw32-384x288/checkpoint_9.pth' \
#     DATASET.TEST_DATASET 'coco' \
#     DATASET.TEST_SET 'val2017' \
#     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/val2017'\
#     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
#     TEST.USE_GT_BBOX True \
#     TEST.BATCH_SIZE_PER_GPU 256 \
#     TEST.POST_PROCESS True \
#     MODEL.NAME 'pose_hrnet_se_lambda' \
#     MODEL.SE_MODULES '[False, False, True, True]' \
#     PRINT_FREQ 100 \
#     TEST.COCO_BBOX_FILE '/home/XYZ/Desktop/datasets/coco/person_detection_results/COCO_test-dev2017_detections_AP_H_609_person.json'


# CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/test_lambda.py \
#     --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
#     GPUS '(0,1,2)' \
#     OUTPUT_DIR 'Outputs/outputs/lambda/lambda_real/' \
#     LOG_DIR 'Outputs/logs/lambda/lambda_real/' \
#     TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/lambda/hrnetw32-256x192/checkpoint_186.pth' \
#     DATASET.TEST_DATASET 'coco' \
#     DATASET.TEST_SET 'val2017' \
#     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/val2017'\
#     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
#     TEST.USE_GT_BBOX True \
#     TEST.BATCH_SIZE_PER_GPU 256 \
#     TEST.POST_PROCESS True \
#     MODEL.NAME 'pose_hrnet_se_lambda' \
#     MODEL.SE_MODULES '[True, True, True, True]' \
#     PRINT_FREQ 100 \
#     TEST.COCO_BBOX_FILE '/home/XYZ/Desktop/datasets/coco/person_detection_results/COCO_test-dev2017_detections_AP_H_609_person.json'


# CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/test_lambda.py \
#     --cfg experiments/coco/hrnet/w48_256x192_adam_lr1e-3.yaml \
#     GPUS '(0,1,2)' \
#     OUTPUT_DIR 'Outputs/outputs/lambda/lambda_real/' \
#     LOG_DIR 'Outputs/logs/lambda/lambda_real/' \
#     TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/lambda/hrnetw48-256x192/checkpoint_84.pth' \
#     DATASET.TEST_DATASET 'coco' \
#     DATASET.TEST_SET 'val2017' \
#     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/val2017'\
#     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
#     TEST.USE_GT_BBOX True \
#     TEST.BATCH_SIZE_PER_GPU 256 \
#     TEST.POST_PROCESS True \
#     MODEL.NAME 'pose_hrnet_se_lambda' \
#     MODEL.SE_MODULES '[False, False, True, True]' \
#     PRINT_FREQ 100 \
#     TEST.COCO_BBOX_FILE '/home/XYZ/Desktop/datasets/coco/person_detection_results/COCO_test-dev2017_detections_AP_H_609_person.json'

# CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/test_lambda.py \
#     --cfg experiments/coco/resnet/res50_256x192_d256x3_adam_lr1e-3.yaml \
#     GPUS '(0,1,2)' \
#     OUTPUT_DIR 'Outputs/outputs/lambda/lambda_real/' \
#     LOG_DIR 'Outputs/logs/lambda/lambda_real/' \
#     TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/lambda/resnet50-256x192/checkpoint_92.pth' \
#     DATASET.TEST_DATASET 'coco' \
#     DATASET.TEST_SET 'val2017' \
#     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/val2017'\
#     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
#     TEST.USE_GT_BBOX True \
#     TEST.BATCH_SIZE_PER_GPU 256 \
#     TEST.POST_PROCESS True \
#     MODEL.NAME 'pose_resnet_se_lambda' \
#     MODEL.SE_MODULES '[False, False, True, True]' \
#     PRINT_FREQ 100 \
#     TEST.COCO_BBOX_FILE '/home/XYZ/Desktop/datasets/coco/person_detection_results/COCO_test-dev2017_detections_AP_H_609_person.json'



# CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/test_lambda.py \
#     --cfg experiments/coco/resnet/res101_256x192_d256x3_adam_lr1e-3.yaml \
#     GPUS '(0,1,2)' \
#     OUTPUT_DIR 'Outputs/outputs/lambda/lambda_real/' \
#     LOG_DIR 'Outputs/logs/lambda/lambda_real/' \
#     TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/lambda/resnet101-256x192/checkpoint_101.pth' \
#     DATASET.TEST_DATASET 'coco' \
#     DATASET.TEST_SET 'val2017' \
#     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/val2017'\
#     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
#     TEST.USE_GT_BBOX True \
#     TEST.BATCH_SIZE_PER_GPU 256 \
#     TEST.POST_PROCESS True \
#     MODEL.NAME 'pose_resnet_se_lambda' \
#     MODEL.SE_MODULES '[False, False, True, True]' \
#     PRINT_FREQ 100 \
#     TEST.COCO_BBOX_FILE '/home/XYZ/Desktop/datasets/coco/person_detection_results/COCO_test-dev2017_detections_AP_H_609_person.json'


# CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/test_lambda.py \
#     --cfg experiments/coco/resnet/res152_256x192_d256x3_adam_lr1e-3.yaml \
#     GPUS '(0,1,2)' \
#     OUTPUT_DIR 'Outputs/outputs/lambda/lambda_real/' \
#     LOG_DIR 'Outputs/logs/lambda/lambda_real/' \
#     TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/lambda/resnet152-256x192/checkpoint_69.pth' \
#     DATASET.TEST_DATASET 'coco' \
#     DATASET.TEST_SET 'val2017' \
#     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/val2017'\
#     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
#     TEST.USE_GT_BBOX True \
#     TEST.BATCH_SIZE_PER_GPU 256 \
#     TEST.POST_PROCESS True \
#     MODEL.NAME 'pose_resnet_se_lambda' \
#     MODEL.SE_MODULES '[False, False, True, True]' \
#     PRINT_FREQ 100 \
#     TEST.COCO_BBOX_FILE '/home/XYZ/Desktop/datasets/coco/person_detection_results/COCO_test-dev2017_detections_AP_H_609_person.json'



# CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/test_lambda.py \
#     --cfg experiments/coco/resnet/res50_384x288_d256x3_adam_lr1e-3.yaml \
#     GPUS '(0,1,2)' \
#     OUTPUT_DIR 'Outputs/outputs/lambda/lambda_real/' \
#     LOG_DIR 'Outputs/logs/lambda/lambda_real/' \
#     TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/lambda/resnet50-384x288/checkpoint_109.pth' \
#     DATASET.TEST_DATASET 'coco' \
#     DATASET.TEST_SET 'val2017' \
#     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/val2017'\
#     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
#     TEST.USE_GT_BBOX True \
#     TEST.BATCH_SIZE_PER_GPU 256 \
#     TEST.POST_PROCESS True \
#     MODEL.NAME 'pose_resnet_se_lambda' \
#     MODEL.SE_MODULES '[False, False, True, True]' \
#     PRINT_FREQ 100 \
#     TEST.COCO_BBOX_FILE '/home/XYZ/Desktop/datasets/coco/person_detection_results/COCO_test-dev2017_detections_AP_H_609_person.json'


# CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/test_lambda.py \
#     --cfg experiments/coco/resnet/res101_384x288_d256x3_adam_lr1e-3.yaml \
#     GPUS '(0,1,2)' \
#     OUTPUT_DIR 'Outputs/outputs/lambda/lambda_real/' \
#     LOG_DIR 'Outputs/logs/lambda/lambda_real/' \
#     TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/lambda/resnet101-384x288/checkpoint_104.pth' \
#     DATASET.TEST_DATASET 'coco' \
#     DATASET.TEST_SET 'val2017' \
#     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/val2017'\
#     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
#     TEST.USE_GT_BBOX True \
#     TEST.BATCH_SIZE_PER_GPU 256 \
#     TEST.POST_PROCESS True \
#     MODEL.NAME 'pose_resnet_se_lambda' \
#     MODEL.SE_MODULES '[False, False, True, True]' \
#     PRINT_FREQ 100 \
#     TEST.COCO_BBOX_FILE '/home/XYZ/Desktop/datasets/coco/person_detection_results/COCO_test-dev2017_detections_AP_H_609_person.json'


# CUDA_VISIBLE_DEVICES=0,1,2 python tools/lambda/test_lambda.py \
#     --cfg experiments/coco/resnet/res152_384x288_d256x3_adam_lr1e-3.yaml \
#     GPUS '(0,1,2)' \
#     OUTPUT_DIR 'Outputs/outputs/lambda/lambda_real/' \
#     LOG_DIR 'Outputs/logs/lambda/lambda_real/' \
#     TEST.MODEL_FILE '/home/XYZ/Desktop/intelligentmix/hrnet/models/lambda/resnet152-384x288/checkpoint_96.pth' \
#     DATASET.TEST_DATASET 'coco' \
#     DATASET.TEST_SET 'val2017' \
#     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/val2017'\
#     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
#     TEST.USE_GT_BBOX True \
#     TEST.BATCH_SIZE_PER_GPU 256 \
#     TEST.POST_PROCESS True \
#     MODEL.NAME 'pose_resnet_se_lambda' \
#     MODEL.SE_MODULES '[False, False, True, True]' \
#     PRINT_FREQ 100 \
#     TEST.COCO_BBOX_FILE '/home/XYZ/Desktop/datasets/coco/person_detection_results/COCO_test-dev2017_detections_AP_H_609_person.json'


