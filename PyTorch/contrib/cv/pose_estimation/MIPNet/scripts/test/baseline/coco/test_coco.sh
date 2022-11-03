cd ../../../..

# # # -----------------------------------------------------------
# # # ---------------------------------------
 # CUDA_VISIBLE_DEVICES=0,1,2 python tools/test.py \
 #     --cfg experiments/coco/hrnet/w48_256x192_adam_lr1e-3.yaml \
 #     GPUS '(0,1,2)' \
 #     OUTPUT_DIR 'Outputs/outputs/coco_hrnet' \
 #     LOG_DIR 'Outputs/logs/coco_hrnet' \
 #     TEST.MODEL_FILE 'models/pytorch/pose_coco/pose_hrnet_w48_256x192.pth' \
 #     DATASET.TEST_DATASET 'coco' \
 #     DATASET.TEST_SET 'val2017' \
 #     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/val2017'\
 #     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
 #     DATASET.TEST_DATASET_TYPE 'coco' \
 #     TRAIN.LR_STEP '(70, 100)' \
 #     TEST.USE_GT_BBOX True \
 #     TEST.BATCH_SIZE_PER_GPU 512 \
 #     TEST.POST_PROCESS True \

 # CUDA_VISIBLE_DEVICES=0,1,2 python tools/test.py \
 #     --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml \
 #     GPUS '(0,1,2)' \
 #     OUTPUT_DIR 'Outputs/outputs/coco_hrnet_dark' \
 #     LOG_DIR 'Outputs/logs/coco_hrnet_dark' \
 #     TEST.MODEL_FILE 'models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth' \
 #     DATASET.TEST_DATASET 'coco' \
 #     DATASET.TEST_SET 'val2017' \
 #     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/val2017'\
 #     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
 #     DATASET.TEST_DATASET_TYPE 'coco' \
 #     TEST.USE_GT_BBOX True \
 #     TEST.BATCH_SIZE_PER_GPU 256 \
 #     TEST.POST_PROCESS True \

  # CUDA_VISIBLE_DEVICES=0,1,2 python tools/test.py \
  #    --cfg experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml \
  #    GPUS '(0,1,2)' \
  #    OUTPUT_DIR 'Outputs/outputs/coco_hrnet' \
  #    LOG_DIR 'Outputs/logs/coco_hrnet' \
  #    TEST.MODEL_FILE 'models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth' \
  #    DATASET.TEST_DATASET 'coco' \
  #    DATASET.TEST_SET 'test2017' \
  #    DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/test2017'\
  #    DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
  #    DATASET.TEST_DATASET_TYPE 'coco' \
  #    TEST.USE_GT_BBOX False \
  #    TEST.BATCH_SIZE_PER_GPU 256 \
  #    TEST.POST_PROCESS True \
  #    TEST.COCO_BBOX_FILE '/home/XYZ/Desktop/datasets/coco/person_detection_results/COCO_test-dev2017_detections_AP_H_609_person.json'


 CUDA_VISIBLE_DEVICES=0,1,2 python tools/test.py \
     --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml \
     GPUS '(0,1,2)' \
     OUTPUT_DIR 'Outputs/paper_coco' \
     LOG_DIR 'Outputs/logs' \
     TEST.MODEL_FILE 'models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth' \
     DATASET.TEST_DATASET 'coco' \
     DATASET.TEST_SET 'val2017' \
     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/val2017'\
     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
     DATASET.TEST_DATASET_TYPE 'coco' \
     TEST.USE_GT_BBOX True \
     TEST.BATCH_SIZE_PER_GPU 256 \
     TEST.POST_PROCESS True \


 # CUDA_VISIBLE_DEVICES=0,1,2 python tools/test.py \
 #     --cfg experiments/coco/resnet/res152_256x192_d256x3_adam_lr1e-3.yaml \
 #     GPUS '(0,1,2)' \
 #     OUTPUT_DIR 'Outputs/outputs/coco_hrnet' \
 #     LOG_DIR 'Outputs/logs/coco_hrnet' \
 #     TEST.MODEL_FILE 'models/pytorch/pose_coco/pose_resnet_152_256x192.pth' \
 #     DATASET.TEST_DATASET 'coco' \
 #     DATASET.TEST_SET 'val2017' \
 #     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/val2017'\
 #     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
 #     DATASET.TEST_DATASET_TYPE 'coco' \
 #     TRAIN.LR_STEP '(70, 100)' \
 #     TEST.USE_GT_BBOX True \
 #     TEST.BATCH_SIZE_PER_GPU 512 \
 #     TEST.POST_PROCESS True \


 # CUDA_VISIBLE_DEVICES=0,1,2 python tools/test.py \
 #     --cfg experiments/coco/resnet/res50_384x288_d256x3_adam_lr1e-3.yaml \
 #     GPUS '(0,1,2)' \
 #     OUTPUT_DIR 'Outputs/outputs/coco_hrnet' \
 #     LOG_DIR 'Outputs/logs/coco_hrnet' \
 #     TEST.MODEL_FILE 'models/pytorch/pose_coco/pose_resnet_50_384x288.pth' \
 #     DATASET.TEST_DATASET 'coco' \
 #     DATASET.TEST_SET 'val2017' \
 #     DATASET.TEST_IMAGE_DIR '/home/XYZ/Desktop/datasets/coco/images/val2017'\
 #     DATASET.TEST_ANNOTATION_FILE '/home/XYZ/Desktop/datasets/coco/annotations/person_keypoints_val2017.json' \
 #     DATASET.TEST_DATASET_TYPE 'coco' \
 #     TRAIN.LR_STEP '(70, 100)' \
 #     TEST.USE_GT_BBOX True \
 #     TEST.BATCH_SIZE_PER_GPU 256 \
 #     TEST.POST_PROCESS True \




