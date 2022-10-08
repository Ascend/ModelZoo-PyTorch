# Yolov4 模型使用说明

## Requirements
* NPU配套的run包安装
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)
* pillow建议安装较新版本，与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision 建议：Pillow版本是9.1.0 torchvision版本是0.6.0
* (可选)参考《Pytorch 网络模型移植&训练指南》6.4.2章节，配置cpu为性能模式，以达到模型最佳性能；不开启不影响功能。

## Dataset Prepare
1. 下载coco数据集，包含图片、annotations、labels
    图片、annotations: 从coco官方网站获取
    labels: https://drive.google.com/uc?export=download&id=1cXZR_ckHki6nddOmcysCuuJFM--T-Q6L
2. 将coco数据集放于工程根目录下
    coco目录结构如下：
	```
    coco
       |-- annotations
       |-- images
          |-- train2017
          |-- val2017   
       |-- labels
          |-- train2017
          |-- val2017
	```	  
## 安装依赖
pip3.7 install -r requirements.txt


## Train Model
### 单卡

```
bash test/train_full_1p.sh  --data_path=coco数据集路径               #单卡精度训练
```

### 8卡
        
```
bash test/train_full_8p.sh  --data_path=coco数据集路径               #8卡精度训练
```
