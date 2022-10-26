### 一、依赖

* NPU配套的run包安装
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)
* torch(NPU版本)
* torchvision
* dllogger

注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision
    建议：Pillow版本是9.1.0  torchvision版本是0.6.0



### 二、训练流程：

*单卡训练流程：

	1.安装环境
	2.开始训练
              bash ./test/train_full_1p.sh  --arch=模型名称 --data_path=imagenet数据集路径  --device_id=NPU卡ID 
*多卡训练流程

	1.安装环境
	2.开始训练
              bash ./test/train_full_8p.sh  --arch=模型名称 --data_path=imagenet数据集路径 
注：arch选resnet1001或resnet1202，默认为resnet1001


### 三、测试结果

 训练日志路径:

        /.../ResNet1001_1202_for_PyTorch/test/output/device_id/


