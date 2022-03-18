# TResNet

This implements training of TResNet on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/rwightman/pytorch-image-models/tree/v0.4.5).

## TResNet Detail

First, Ascend-Pytorch has not implement a third-party inplace-abn, so we have to use BatchNorm2D to replace it.
Second, for op PadV3D,reflect mode is not supported currently,so we replace it with Conv2D's padding.
And then, as of the current date, Ascend-Pytorch is still inefficient for contiguous operations.Therefore, TResNet is using whilte shape list to solve the problem about TransposeD.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset 
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Before Traing

Please add these shape into white shape list before training:[ 190,3, 224, 224],[ 190, 3, 56, 4, 56, 4],[ 190, 4, 4, 3, 56, 56],

## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
# training 1p performance
bash ./test/train_performance_1p.sh --data_path=${real_data_path}

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=${real_data_path}

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=${real_data_path}

#test 1p accuracy
bash test/train_eval_1p.sh --data_path=${real_data_path}/val --pth_path=${real_pre_train_model_path}

# finetuning 1p 
bash test/train_finetune_1p.sh --data_path=${real_data_path} --pth_path=${real_pre_train_model_path} --num_classes=1001

# demo
python demo.py ${real_pre_train_model_path}
```

Log path:
    log/train_log_8p          # training detail log
    log/perf_8p.log			  # 8p training performance result log
    log/eval.log   			  # 8p training accuracy result log



## TResNet training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 927       | 1        | 1        | O2       |
| 78.87    | 6200      | 8        | 110      | O2       |