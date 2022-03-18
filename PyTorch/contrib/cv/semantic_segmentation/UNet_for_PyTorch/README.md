# UNet

This implements training of UNet on the 2018 Data Science Bowl dataset, mainly modified from [UNet](https://github.com/4uiiurz1/pytorch-nested-unet).

## UNet Detail 

For details, see [UNet](https://github.com/4uiiurz1/pytorch-nested-unet).


## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- pip install -r requirements.txt
- get dataset from [data-science-bowl-2018](https://www.kaggle.com/c/data-science-bowl-2018/data).The file structure is the following:
```
inputs
└── data-science-bowl-2018
    ├── stage1_train
    |   ├── 00ae65...
    │   │   ├── images
    │   │   │   └── 00ae65...
    │   │   └── masks
    │   │       └── 00ae65...            
    │   ├── ...
    |
    ...
```
- the data-science-bowl-2018 dataset need preprocess. 
```bash
python3.7.5 preprocess_dsb2018.py
```

## Training

- 1p training 1p 
    - bash ./test/train_full_1p.sh --data_path=xxx # training accuracy

    - bash ./test/train_performance_1p.sh --data_path=xxx # training performance

- 8p training 8p
    - bash ./test/train_full_8p.sh --data_path=xxx # training accuracy
    
    - bash ./test/train_performance_8p.sh --data_path=xxx # training performance

- eval default 8p， should support 1p
    - bash ./test/train_eval_8p.sh --data_path=xxx

- Traing log
    - test/output/devie_id/train_${device_id}.log # training detail log
    
    - test/output/devie_id/ResNet101_${device_id}_bs_8p_perf.log # 8p training performance result
    
    - test/output/devie_id/ResNet101_${device_id}_bs_8p_acc.log # 8p training accuracy result

```
# online inference demo 
python3.7.5 demo.py
(分割后的图片存储在outputs/UNet_Demo/)
# To ONNX
python3.7.5 pthtar2onnx.py
```

## EfficientNet-B3 training result 

| IOU  | FPS  | Npu_nums | Epochs | AMP_Type |
| :----: | :--: | :------: | :----: | :------: |
|   -    | -   |    1     |  100   |    O2    |
| 83.31 | -  |    8     |  100   |    O2    |



