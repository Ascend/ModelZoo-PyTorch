一、依赖
* NPU配套的run包安装
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)
* torch(NPU版本)
* torchvision

注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision
    建议：Pillow版本是9.1.0  torchvision版本是0.6.0

二、 准备数据集
1. 下载并解压训练集和测试集图片数据
2. 按顺序对文件夹进行组织
```
./data_dir
./data_dir/ICDAR
./data_dir/ICDAR/Challenge
./data_dir/ICDAR/Challenge/ch4_training_images
./data_dir/ICDAR/Challenge/ch4_training_localization_transcription_gt
./data_dir/ICDAR/Challenge/ch4_test_images
./data_dir/ICDAR/Challenge/ch4_test_localization_transcription_gt
```

三、训练流程：
    
单卡训练流程：

```
	1.安装环境
	2.开始训练
        bash ./test/train_full_1p.sh  --data_path=数据集路径  --device_id=NPU卡ID 
```

	
多卡训练流程

```
	1.安装环境
	2.开始训练
        bash ./test/train_full_8p.sh  --data_path=数据集路径 
```

	
四、Docker容器训练：
    
1.导入镜像二进制包docker import ubuntuarmpytorch.tar REPOSITORY:TAG, 比如:

        docker import ubuntuarmpytorch.tar pytorch:b020

2.执行docker_start.sh后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：

        ./docker_start.sh pytorch:b020 /train/imagenet /home/PSENet

3.执行步骤一训练流程（环境安装除外）
	

五、测试结果
    
训练日志路径:

        /home/PSENet/test/output/device_id/