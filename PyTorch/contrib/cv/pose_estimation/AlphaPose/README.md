# AlphaPose

This implements training of AlphaPose on the COCO dataset, mainly modified from [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)

## AlphaPose Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. Therefore, AlphaPose is re-implemented using semantics such as custom OP. For details, see alphapose/models/fastpose.py . 

## Requirements

-   install Ascend-Pytorch 

-   install apex 

-   install related lib
      ```
    #ubuntu 
    apt-get install libyaml-dev
    #centOS 
    yum install libyaml-devel
      ```

-   install alphapose

    ```
    python setup.py build develop
    ```

## Before Training

1.  Prepare COCO datasets

    ```
    |-- coco
        `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- train2017
        |   |-- 000000000009.jpg
        |   |-- 000000000025.jpg
        |   |-- 000000000030.jpg
        |   |-- ... 
        `-- val2017
        |-- 000000000139.jpg
        |-- 000000000285.jpg
        |-- 000000000632.jpg
        |-- ... 
    ```
2.  Modify datasets path config` ROOT:'/home/dataset/coco2017'` in /configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml

    ```
    DATASET:
      TRAIN:
        TYPE: 'Mscoco'
        ROOT: '/home/dataset/coco2017/'
        IMG_PREFIX: 'train2017'
        ANN: 'annotations/person_keypoints_train2017.json'
        AUG:
          FLIP: true
          ROT_FACTOR: 40
          SCALE_FACTOR: 0.3
          NUM_JOINTS_HALF_BODY: 8
          PROB_HALF_BODY: -1
      VAL:
        TYPE: 'Mscoco'
        ROOT: '/home/dataset/coco2017/'
        IMG_PREFIX: 'val2017'
        ANN: 'annotations/person_keypoints_val2017.json'
      TEST:
        TYPE: 'Mscoco_det'
        ROOT: '/home/dataset/coco2017/'
        IMG_PREFIX: 'val2017'
        DET_FILE: './exp/json/test_det_yolo.json'
        ANN: 'annotations/person_keypoints_val2017.json'
      DEMO:
        TYPE: 'Mscoco_Infer'
        ROOT: '/home/dataset/coco2017/'
        IMG_PREFIX: 'val2017'
        ANN: 'annotations/person_keypoints_val2017.json'
    ``

## Training & Inference

# 1p training
bash test/train_full_1p.sh        # training full epoches
bash test/train_performance_1p.sh # training one epoch to see performance

# 8p training
bash test/train_full_8p.sh        # training full epoches
bash test/train_performance_8p.sh # training one epoch to see performance

# eval default 8p
bash test/train_eval_8p.sh

# Online inference demo
python scripts/demo.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint exp/exp_test-256x192_res50_lr1e-3_1x.yaml/model_199.pth

# To ONNX
python scripts/pthtar2onnx.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml

## AlphaPose training result

|        | gt mAP | Epochs | AMP_Type |
|:------:|:------:|:------:|:--------:|
| 1p-GPU |   -    |  200   |    O2    |
| 1p-NPU |   -    |  200   |    O2    |
| 8p-GPU | 72.24  |  200   |    O2    |
| 8P-NPU | 71.61  |  200   |    O2    |

# Statement
For details about the public address of the code in this repository, you can get from the file public_address_statement.md
