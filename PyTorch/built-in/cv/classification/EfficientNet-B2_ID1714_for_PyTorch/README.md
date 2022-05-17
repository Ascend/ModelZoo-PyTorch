一、训练流程：
    
单卡训练流程：

```
	1.安装环境
	2.修改train_performance_1p.sh字段"data"为当前磁盘的数据集路径
	3.修改字段device_id（单卡训练所使用的device id），为训练配置device_id，比如device_id=0
	4.bash ./test/train_full_1p.sh  --data_path=数据集路径           # 精度训练
```

	
多卡训练流程

```
	1.安装环境
	2.修改多P脚本中字段"data"为当前磁盘的数据集路径
	3.修改train_performance_8p.sh字段"addr"为当前主机ip地址
	4.bash ./test/train_full_8p.sh  --data_path=数据集路径           # 精度训练
```




	
二、Docker容器训练：
    
1.导入镜像二进制包docker import ubuntuarmpytorch.tar REPOSITORY:TAG, 比如:

        docker import ubuntuarmpytorch.tar pytorch:b020

2.执行docker_start.sh后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：

        ./docker_start.sh pytorch:b020 /train/imagenet /home/Efficientnet

3.执行步骤一训练流程（环境安装除外）
	
三、测试结果
    
训练日志路径：在训练脚本的同目录下result文件夹里，如：

         /home/Efficientnet/test/output/0/train_0.log
        
        
	

