# RefineNet

This repository is an NPU implementation of the ["RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation"](https://arxiv.org/abs/1611.06612), referring to https://github.com/DrSleep/refinenet-pytorch



## Requirements

See requirements.txt

- PyTorch 
- torchvision 
- Numpy 1.15.1
- Pillow 9.1.0
- h5py 2.8.0
- tqdm 4.28.1
- h5py 3.4.0
- opencv-python 3.4.4.19
- albumentations 0.4.5
  Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
- install densetorch as follow:

```bash
 git clone https://github.com/DrSleep/DenseTorch
 cd ./DenseTorch
 python setup.py install
```

## Training

The processed VOC dataset can be downloaded from [Download](https://pan.baidu.com/s/12wHGhby5vEcG6isQpnpcMQ) with extraction code: vnhb (about 9 G), put it in ./VOC. Or, you can download it by:
```bash
bash load_data.sh
```

The training common: 

```bash
# 1p train perf
bash test/train_performance_1p.sh 

# 8p train perf
bash test/train_performance_8p.sh

# 8p train full
bash test/train_full_8p.sh

# finetuning
bash test/train_finetune_1p.sh
```

In the first running, it requires time to downloaded the model pre-trained on ImageNet. Or you can manually download it by:
```shell
cd ~
mkdir .torch
mkdir .torch/models
cd .torch/models
wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth 
mv resnet101-5d3b4d8f.pth 101_imagenet.pth.tar
```
Log path: ./log/

Saved  model path: ./model/

## Training result

| IOU | FPS | NPU_nums | BS/NPU | AMP_type |
|-----------|-------|-------|-----------------|-----------|
| 78.56 | 25.56 | 1 | 16 | O2 |
| 77.34 | 159.46| 8 | 8 | O2 |




## Citation
```
RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation
Guosheng Lin, Anton Milan, Chunhua Shen, Ian Reid
In CVPR 2017
```

## Statement
```
For details about the public address of the code in this repository, you can get from the file public_address_statement.md
```