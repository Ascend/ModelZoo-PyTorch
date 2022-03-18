# Yolov4 模型使用说明

## Requirements
* NPU配套的run包安装
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)
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
1. 运行 train_1p.sh
```
chmod +x train_1p.sh
./train_1p.sh
```
若需要指定训练使用的卡号, 可修改train_1p.sh文件 "--device_id 0"配置项,其中卡号为0-7

### 8卡
1. 运行 train_8p.sh
```
chmod +x train_8p.sh
./train_8p.sh
```

## 验证
复制训练好的last.pt到weights文件夹下，运行test.sh
```
chmod +x test.sh
./test.sh
```