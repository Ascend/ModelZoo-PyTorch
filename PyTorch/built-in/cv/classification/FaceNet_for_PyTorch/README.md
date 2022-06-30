FaceNet模型使用说明

## Requirements
* NPU配套的run包安装
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)
* 还需安装（NPU-driver.run, NPU-firmware.run, NPU-toolkit.run)
* (可选)参考《Pytorch 网络模型移植&训练指南》6.4.2章节，配置cpu为性能模式，以达到模型最佳性能；不开启不影响功能。
注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision 
建议：Pillow版本是9.1.0 torchvision版本是0.6.0

## Dataset Prepare
1. 下载VGGFace2数据集
2. 使用MTCNN网络对数据集进行预处理，预处理后会在数据集路径下生成“train_cropped/”目录。

## 预训练模型下载
1. 参考model/inception_resnet_v1.py,下载对应预训练模型
2. 若无法自动下载，可手动下载模型，并放到/root/.cache/torch/checkpoints/文件夹下。


### Build FaceNet from source
1. 下载modelzoo项目zip文件并解压
2. 压缩modelzoo\built-in\PyTorch\Official\cv\image_classification\FaceNet+for_PyTorch目录
3. 于npu服务器解压FaceNet_for_PyTorch压缩包


## Train MODEL

### 单卡
bash ./test/train_full_1p.sh  --data_path=数据集路径                    # 精度训练

### 8卡
bash ./test/train_full_8p.sh  --data_path=数据集路径                    # 精度训练