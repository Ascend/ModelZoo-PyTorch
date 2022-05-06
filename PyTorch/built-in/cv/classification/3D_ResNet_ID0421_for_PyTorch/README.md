环境
----------
    torchvision==0.2.2.post2
    h5py
    FFmpeg
    FFprobe
    tensorboard
    tensorboard
## Training

一、训练流程

单卡训练流程：

    1.安装环境  
    2.开始训练
        bash ./test/train_full_1p.sh  --data_path=数据集路径          # 精度训练
        bash ./test/train_performance_1p.sh  --data_path=数据集路径    # 性能训练


​    
多卡训练流程

    1.安装环境
    2.开始训练
        bash ./test/train_full_8p.sh  --data_path=数据集路径     # 精度训练
        bash ./test/train_performance_8p.sh  --data_path=数据集路径    # 性能训练

注：脚本模型使用的是hmdb51数据集， --data_path的值为数据集目录

二、Docker容器训练

1.导入镜像二进制包docker import ubuntuarmpytorch.tar REPOSITORY:TAG, 比如:

    docker import ubuntuarmpytorch.tar pytorch:b020

2.执行docker_start.sh后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：


    ./docker_start.sh pytorch:b020  /datapt_resnet3d_data  /home/3D_ResNet_ID0421_for_PyTorch

3.执行步骤一训练流程（环境安装除外）

三、训练结果
/home/3D_ResNet_ID0421_for_PyTorch/output/0/