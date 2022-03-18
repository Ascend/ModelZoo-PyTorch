FaceNet模型使用说明

## Requirements
* NPU配套的run包安装
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)
* 还需安装（NPU-driver.run, NPU-firmware.run, NPU-toolkit.run)
* (可选)参考《Pytorch 网络模型移植&训练指南》6.4.2章节，配置cpu为性能模式，以达到模型最佳性能；不开启不影响功能。


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
1. 运行 run_1p.sh (其中在run_1p.sh脚本中“--device_list”参数可以设置开启第几张卡）
```
sh run_1p.sh
```

### 8卡
1. 运行 run_8p.sh
```
sh run_8p.sh
```