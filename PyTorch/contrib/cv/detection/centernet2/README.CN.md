# CenterNet2

本项目实现了 CenterNet2 在 NPU 上的训练.
[CenterNet2 github链接](https://github.com/xingyizhou/CenterNet2)

## 1.CenterNet2 Detail

本项目对 CenterNet2 做了如下更改：
1. 迁移到 NPU 上
2. 使用混合精度训练、测试
3. 对于一些操作，固定动态 shape 、使用 NPU 算子优化性能、同时将一些操作转移到 CPU 上进行


## 2.Requirements
### 2.1 安装NPU软件

* NPU配套的run包安装
* PyTorch(NPU版本)
* apex(NPU版本)

### 2.2 安装第三方软件

(1) 通过pip 安装部分第三方软件：

```
pip3 install -r requirements.txt
```

(2) 安装opencv

```
pip install opencv-python
```

(3) 编译安装torchvision

```
git clone https://github.com/pytorch/vision
cd vision
git checkout v0.6.0
python3 setup.py install
```

**注：您必须编译安装torchvision，直接使用pip安装训练将无法启动**

(4) 编译detectron2

进入模型脚本根目录，编译detectron2

```
python3 setup.py build develop
```


(5) 下载预训练模型 R-50.pkl ,projects/CenterNet2/configs/Base-CenterNet2.yaml配置文件中MODEL.WEIGHTS 设置为R-101.pkl的绝对路径
## 3.Dataset

(1) 下载coco2017数据集；

(2) 解压数据集，解压后的目录结构如下
```
│   ├── coco
│       ├── annotations
│       ├── train2017
│       ├── val2017
```
(3) 通过设置环境变量DETECTRON2_DATASETS=“coco 所在数据集路径”进行设置，如 export DETECTRON2_DATASETS=/home/，则 coco 数据集放在 /home/ 目录中

## 4.Training

### 4.1 NPU 1P

在模型根目录下，运行 train_full_1p.sh，同时传入参数--data_path，指定为coco数据集的路径父路径(例如数据集路径为/home/coco,则--data_path=/home)

```
bash ./test/train_full_1p.sh --data_path=/home
```
模型训练结束后，会在result/CenterNet2/CenterNet2_R50_1x目录下保存模型文件model_final.pth，训练结束后若要评估精度需运行eval脚本，参考第6节

### 4.2 NPU 8P

在模型根目录下，运行 train_full_8p.sh，同时传入参数--data_path，指定为coco数据集的路径父路径(例如数据集路径为/home/coco,则--data_path=/home)

```
bash ./test/train_full_8p.sh --data_path=/home
```

模型训练结束后，会在result/CenterNet2/CenterNet2_R50_1x目录下保存模型文件model_final.pth，训练结束后若要评估精度需运行eval脚本，参考第6节

## 5.Finetune 

请将projects/CenterNet2/configs/Base-CenterNet2.yaml中的字段WEIGHTS修改为第3节中model_final.pth的绝对路径，如修改为
```
WEIGHTS: "/home/CenterNet2/result/CenterNet2/CenterNet2_R50_1x/model_final.pth"
```
然后启动评估脚本

```
bash ./test/train_finetune_1p.sh --data_path=/home
```

## 6.评估
训练结束后，需要手动启动评估程序。请将projects/CenterNet2/configs/Base-CenterNet2.yaml中的字段WEIGHTS修改为第3节中model_final.pth的绝对路径，如修改为
```
WEIGHTS: "/home/CenterNet2/result/CenterNet2/CenterNet2_R50_1x/model_final.pth"
```

然后启动评估脚本

```
bash ./test/train_eval_1p.sh --data_path=/home
```
## CenterNet2 Result

| 名称   | 精度  | 性能     |
| ------ | ----- | -------- |
| GPU-1p | -     | 12.3fps  |
| NPU-1p | -     | 2.86 fps |
| GPU-8p | 43.68 | 90.5fps  |
| NPU-8p | 43.5  | 18.7fps  |