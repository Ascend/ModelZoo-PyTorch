# YOLACT++

This implements training of Yolact++ on the COCO2017 dataset, mainly modified from [yolact++](https://github.com/dbolya/yolact).

## YOLACT++ Detail

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the dataset by
- `sh data/scripts/COCO.sh`
- Download  an imagenet-pretrained model and put it in ./weights.
    - For Resnet101, download resnet101_reducedfc.pth.
    - For Resnet50, download resnet50-19c8e357.pth.
    - For Darknet53, download darknet53.pth.

## Training

To train a model, run `train.py` with the desired model architecture and the path to dataset:

```bash
# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=data_path

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=data_path

# training 8p accuracy
bash ./test/train_full_8p.sh  --data_path=data_path

# training 8p performance
bash ./test/train_performance_8p.sh  --data_path=data_path

#test 8p accuracy
bash test/train_eval_1p.sh --$pth_path=pth_path
```

Log path:
    ${YOLACT_ROOT}/test/output/0/train_0.log      # training detail log



## Yolact++ training result

| 名称    | 精度       | FPS | AMP_Type   |
| :------: | :------:  | :------: | :------: |
| NPU-1p        |  -      |   3.153      |   O0      |
|  NPU-8p  |   33.49    |    14.677     |   O0    |
