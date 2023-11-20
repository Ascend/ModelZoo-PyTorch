# Pix2PixHD

本项目实现了 Pix2PixHD 在 NPU 上的训练.
[Pix2PixHD github链接](https://github.com/NVIDIA/pix2pixHD)

## 1.Pix2PixHD Detail

本项目对 Pix2PixHD 做了如下更改：
1. 迁移到 NPU 上
2. 使用混合精度训练
3. 在数据预处理阶段将NPU不支持的操作转移到 CPU 上进行


## 2.Requirements
### 2.1 安装NPU软件

* NPU配套的run包安装
* PyTorch(NPU版本)
* apex(NPU版本)

### 2.2 安装第三方软件

(1) 通过pip 安装部分第三方软件：

```
pip install -r requirements.txt
```
注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision ,建议Pillow版本是9.1.0 torchvision版本是0.6.0
## 3.Dataset

(1) 下载[cityscapes](https://www.cityscapes-dataset.com/downloads/)数据集，本项目只需要下载cityscapes数据集中的gtFine_trainvaltest.zip和leftImg8bit_trainvaltest.zip两个压缩包；

(2) 解压gtFine_trainvaltest.zip和leftImg8bit_trainvaltest.zip两个压缩包，解压后的目录结构如下(假定解压后的目录位于/home目录下)
```
│   ├── home
│       ├── gtFine
│         ├── test
|         ├── train
|         ├── val
│       ├── leftImg8bit
|         ├── test
|         ├── train
|         ├── val
```
(3) 通过设置环境变量DATASETS=“cityscapes 所在数据集路径”进行设置，如 export DATASETS=/home/，则 cityscapes 数据集放在 /home/ 目录中

(4) 运行python datasets_deal.py，将下载得到的数据集转换为本项目需要的数据组织形式，转换之后的数据集目录如下
```
|    ├── home
│       ├── cityscapes
│           ├── train_img
│           ├── train_label
|           ├── train_inst
```
**注意：datasets_deal.py仅仅将压缩包中的训练集相关的图像整理为以上目录格式，并未对压缩包中的测试集图像做处理,测试图像采用pix2pixHD的github原仓中提供的测试图片**

(5)在主目录下创建/datasets/cityscapes目录，并将整理好的数据集train_img、train_label、train_inst三个目录复制到本项目的主目录下的datasets/cityscapes下
```
mkdir -p datasets/cityscapes
mv /home/cityscapes/train_img ./datasets/cityscapes
mv /home/cityscapes/train_label ./datasets/cityscapes
mv /home/cityscapes/train_inst ./datasets/cityscapes
```

(6)从[Pix2PixHD github链接](https://github.com/NVIDIA/pix2pixHD)下载源代码到/home目录下

```
git clone https://github.com/NVIDIA/pix2pixHD /home/pix2pixHD
```

(7)将pix2pixHD原仓中的测试集复制到本项目的主目录下的datasets/cityscapes下

```
cp -r /home/pix2pixHD/datasets/cityscapes/test_inst ./datasets/cityscapes
cp -r /home/pix2pixHD/datasets/cityscapes/test_label ./datasets/cityscapes
```

(8) 经过以上数据集的处，在本项目的主目录下的datasets目录下有如下的数据集结构
```
|    ├── datasets
│       ├── cityscapes
│           ├── train_img
│           ├── train_label
|           ├── train_inst
│           ├── test_inst
|           ├── test_label
```
**注意：在完成(1)到(7)步，请确认本项目主目录下的datasets目录中是否有(8)的目录结构，当一致时才可进行以下操作，不一致请检查以上(1)到(7)步!**
## 4.Training

### 4.1 NPU 1P

在模型根目录下，运行 train_performance_1p.sh，同时传入参数--data_path，指定为cityscapes数据集的路径父路径(例如数据集路径为./datasets,则--data_path=./datasets)

```
bash ./test/train_performance_1p.sh --data_path=./datasets
```
模型训练结束后，会在./checkpoints/label2city_1024p目录下保存模型文件latest_net_G.pth

### 4.2 NPU 8P

在模型根目录下，运行 train_full_8p.sh，同时传入参数--data_path，指定为cityscapes数据集的路径父路径(例如数据集路径为./datasets,则--data_path=./datasets)

```
bash ./test/train_performance_8p.sh --data_path=./datasets
```

### 4.3 NPU 8p

在模型根目录下，运行 train_performance_8p.sh，同时传入参数--data_path，指定为cityscapes数据集的路径父路径(例如数据集路径为./datasets,则--data_path=./datasets)

```
bash ./test/train_full_8p.sh --data_path=./datasets
```

模型训练结束后，会在./checkpoints/label2city_1024p目录下保存模型文件latest_net_G.pth

### 4.4 NPU 1p

训练结束后，运行train_eval_1p.sh，则会训练好的最新的模型latest_net_G.pth生成测试效果图，测试图片在./datasets/cityscapes/test_inst 和./datasets/cityscapes/test_label，测试结果的输出在./results/label2city_1024p/test_latest/img下。

```
bash ./test/train_eval_1p.sh --data_path=./datasets
```

**注意：本模型单卡训练效果比8卡训练效果要好，单卡可获得最好的训练效果！**
## Pix2PixHD Result

| 名称   | 精度  |  性能     |
| ------ | ----- | --------  |
| GPU-1p | -     | 4.55 fps  |
| NPU-1p | -     | 3.76 fps  |
| GPU-8p | -     | 19.14 fps |
| NPU-8p | -     | 13.79 fps |


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md