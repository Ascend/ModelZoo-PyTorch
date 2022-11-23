# Efficientnet-B0

This implements training of Efficientnet-B0 on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).

## Efficientnet-B0 Detail

1. 迁移到torch1.8
3. 原README.md更名为README_raw.md

## Requirements

- `pip install -r requirements.txt`
- apex & torch version：
  - apex：0.1+ascend.20220620
  - torch：1.8.1+ascend.rc2.20220620
  - torch-npu：1.8.1rc2.post20220620
- Download the ImageNet dataset
  - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

- To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:
- `real_data_path` is a directory containing `{train, val}` sets of ImageNet
- For 8p eval,  `resume_pth`  is the path to the model you want to evaluate.

```bash
# training 1p performance，单p上运行1个epoch，运行时间约为40min，
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training 8p performance，8p上运行1个epoch，运行时间约为30min
bash ./test/train_performance_8p.sh --data_path=real_data_path

# training 8p full，8p上运行100个epoch，运行时间约为12h
bash ./test/train_full_8p.sh --data_path=real_data_path

```

- 注：在1p脚本`./test/train_performance_1p.sh` 中，可以修改变量`device_id`，指定NPU训练，默认`device_id=0`

## Efficientnet-B0 Training Result

| Acc@1  |  FPS   | Npu_nums | Epochs | AMP_Type | Torch |
| :----: | :----: | :------: | :----: | :------: | :---: |
|   -    | 1307.3 |    1     |   1    |    O1    |  1.8  |
| 74.219 | 2703.8 |    8     |  100   |    O1    |  1.8  |