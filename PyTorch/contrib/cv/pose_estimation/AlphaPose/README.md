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
    ```
3.  Set env (This is may different from the example in official website)

        export LD_LIBRARY_PATH=/usr/local/:/usr/local/lib/:/usr/lib64/:/usr/lib/:/usr/local/python3.7.5/lib/:/usr/local/openblas/lib:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
        export PATH=$PATH:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:/usr/local/Ascend/ascend-toolkit/latest/toolkit/tools/ide_daemon/bin/
        export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/
        export OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/plugin/opskernel/libfe.so:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/plugin/opskernel/libaicpu_engine.so:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/plugin/opskernel/libge_local_engine.so
        export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/python/site-packages/:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH
        
        ulimit -SHn 51200
        export DYNAMIC_OP="ADD#MUL"
        export TASK_QUEUE_ENABLE=0
        export ASCEND_SLOG_PRINT_TO_STDOUT=0
        export ASCEND_GLOBAL_LOG_LEVEL=3
        export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
        
        path_lib=$(python3.7 -c """ 
        import sys
        import re
        result=''
        for index in range(len(sys.path)):
            match_sit = re.search('-packages', sys.path[index])
            if match_sit is not None:
                match_lib = re.search('lib', sys.path[index])
                if match_lib is not None:
                    end=match_lib.span()[1]
                    result += sys.path[index][0:end] + ':'
                result+=sys.path[index] + '/torch/lib:'
        print(result)"""
        )
        export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib/:${path_lib}:$LD_LIBRARY_PATH

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
```

## AlphaPose training result

|        | gt mAP | Epochs | AMP_Type |
| :------: | :------: | :------: | :--------: |
| 1p-GPU | -     | 200    | O2       |
| 1p-NPU | -      | 200    | O2       |
| 8p-GPU | 72.24  | 200    | O2       |
| 8P-NPU | 71.61  | 200    | O2       |

