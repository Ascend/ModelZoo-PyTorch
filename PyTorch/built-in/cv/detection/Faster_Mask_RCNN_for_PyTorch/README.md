# 目标检测与目标分割类模型使用说明

## Requirements
    * NPU配套的run包安装(C20B030)
    * Python 3.7.5
    * PyTorch(NPU版本)
    * apex(NPU版本)
    注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision,建议Pillow版本是9.1.0 torchvision版本是0.6.0
## Dataset Prepare
1. 下载COCO数据集

## Pre-Train Weights File 
1. 模型脚本会自动下载预训练权重文件。若下载失败，请自行准备R-101.pkl等权重文件，将文件放到数据集同级路径下。

### Build Detectron2 from Source

编译器版本：gcc & g++ ≥ 5
```
python3.7 -m pip install -e Faster_Mask_RCNN_for_PyTorch

```
在重装PyTorch之后，通常需要重新编译detectron2。重新编译之前，需要使用`rm -rf build/ **/*.so`删除旧版本的build文件夹及对应的.so文件。


##mask_rcnn启动训练 

    单卡
         bash ./test/train_full_1p.sh  --data_path=数据集路径  

    8卡
         bash ./test/train_full_8p.sh  --data_path=数据集路径  

##faster_rcnn启动训练 

    单卡
         bash ./test/train_faster_rcnn_full_1p.sh  --data_path=数据集路径  

    8卡
         bash ./test/train_faster_rcnn_full_8p.sh  --data_path=数据集路径 

## Docker容器训练：

1.导入镜像二进制包docker import ubuntuarmpytorch_maskrcnn.tar REPOSITORY:TAG, 比如:

    docker import ubuntuarmpytorch_maskrcnn.tar pytorch:b020_maskrcnn
2.执行docker_start.sh 后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：

    ./docker_start.sh pytorch:b020_maskrcnn /train/coco /home/MaskRCNN

