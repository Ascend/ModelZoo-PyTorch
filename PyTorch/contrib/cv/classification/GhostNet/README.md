# GhostNet 1.0× 

This implements training of GhostNet 1.0× on the ImageNet dataset, mainly modified from https://github.com/rwightman/pytorch-image-models https://github.com/huawei-noah/ghostnet 

## GhostNet 1.0× Detail 

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 
Therefore, GhostNet 1.0× is re-implemented using semantics such as custom OP. For details, seepytorch-image-models/ghostnet/ghostnet_pytorch/ghostnet.py 


## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- pip install timm
- pip install git+https://github.com/rwightman/pytorch-image-models.git
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training 


```bash
# 1p train 1p
bash test/train_full_1p.sh --data_path={data/path} # train accuracy

bash test/train_performance_1p.sh --data_path={data/path} # train performance

#  8p train 8p
bash test/train_full_8p.sh --data_path={data/path} # train accuracy

bash test/train_performance_8p.sh --data_path={data/path} # train performance

# 1p eval 1p
bash test/train_eval_1p.sh --data_path={data/path}

# online inference demo 
python3.7 demo.py

```

## GhostNet 1.0× training result 

|  Acc@1  |   FPS    | Npu_nums | Epochs | AMP_Type |  Device_Type   |
| :-----: | :------: | :------: | :----: | :------: |  :---------:   |
|    -    | 1378.8   |    1     |  10    |    O2    | 910B_Pro_aarch |
| 73.129  | 9559.2   |    8     |  400   |    O2    | 910B_Pro_aarch |



