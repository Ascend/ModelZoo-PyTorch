一、依赖
* NPU配套的run包安装
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)
* torch(NPU版本)
* torchvision
* dllogger

注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision
    建议：Pillow版本是9.1.0  torchvision版本是0.6.0

二、训练流程：
    
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

多机多卡性能数据获取流程

```
	1.安装环境
	2.开始训练，每个机器所请按下面提示进行配置
              bash ./test/train_performance_multinodes.sh  --data_path=数据集路径 --batch_size=单卡batch_size*所有卡数 --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
```


	
三、Docker容器训练：
    
1.导入镜像二进制包docker import ubuntuarmpytorch.tar REPOSITORY:TAG, 比如:

        docker import ubuntuarmpytorch.tar pytorch:b020

2.执行docker_start.sh后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：

        ./docker_start.sh pytorch:b020 /train/imagenet /home/ResNet50

3.执行步骤一训练流程（环境安装除外）
	

四、测试结果
    
训练日志路径:

        /home/ResNet50/test/output/device_id/
	
Note: 本模型单卡和多卡使用不同的脚本，脚本配置有差异， 会影响到线性度， 目前正在重构中；
