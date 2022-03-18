# ResNet152_ID0424_for_PyTorch

This implements training of ResNet152_ID0424_for_PyTorch on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).


## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training 

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash

#全量精度执行
cd test
bash train_full_8p.sh --data_path=/npu/traindata/imagenet_pytorch

#性能执行命令
cd test
bash train_performance_8p.sh --data_path=/npu/traindata/imagenet_pytorch
```

## ResNet152_ID0424_for_PyTorch training result 
| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 698       | 1        | 110      | O2       |
| 76.95    | 3687      | 8        | 110      | O2       |