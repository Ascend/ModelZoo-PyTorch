一、依赖
* NPU配套的run包安装
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)
* torch(NPU版本)
* torchvision
* dllogger

二、训练流程：
    
单卡训练流程：

```
	1.安装环境
	2.修改train_performance_1p.sh字段"data"为当前磁盘的数据集路径；
	3.修改字段device_id（单卡训练所使用的device id），为训练配置device_id，比如device_id=0；
	4.执行bash train_performance_1p.sh单卡脚本， 进行单卡训练；
```

	
多卡训练流程

```
	1.安装环境
	2.修改多P脚本中字段"data"为当前磁盘的数据集路径
	3.修改字段device_id_list（多卡训练所使用的device id列表），为训练配置device_id，比如4p,device_id_list=0,1,2,3；8P默认使用0，1，2，3，4，5，6，7卡不用配置
	4.执行bash train_performance_8p.sh等多卡脚本， 进行多卡训练;
```



	
三、Docker容器训练：
    
1.导入镜像二进制包docker import ubuntuarmpytorch.tar REPOSITORY:TAG, 比如:

        docker import ubuntuarmpytorch.tar pytorch:b020

2.执行docker_start.sh后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：

        ./docker_start.sh pytorch:b020 /train/imagenet /home/ResNet50

3.执行步骤一训练流程（环境安装除外）
	

四、测试结果
    
训练日志路径：在训练脚本的同目录下result文件夹里，如：

        /home/ResNet50/test/output/device_id/training_8p_job_20201121023601
	

