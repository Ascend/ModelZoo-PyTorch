环境
----------
    pytorch 1.5
    torchvision 0.5.0
    apex 0.1
    easydict 1.9
    lmdb 0.98
    PyYAML 5.3
## Training

一、训练流程

单卡训练流程：

    1.安装环境  
    2.开始训练
        bash ./test/train_full_1p.sh  --data_path=数据集路径  --epochs=训练周期           # 精度训练
        bash ./test/train_performance_1p.sh  --data_path=数据集路径  --epochs=训练周期    # 性能训练


​    
多卡训练流程

    1.安装环境
    2.开始训练
        bash ./test/train_full_8p.sh  --data_path=数据集路径  --epochs=训练周期           # 精度训练
        bash ./test/train_performance_8p.sh  --data_path=数据集路径  --epochs=训练周期    # 性能训练


二、Docker容器训练

1.导入镜像二进制包docker import ubuntuarmpytorch.tar REPOSITORY:TAG, 比如:

    docker import ubuntuarmpytorch.tar pytorch:b020

2.执行docker_start.sh后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：


    ./docker_start.sh pytorch:b020 /train/peta /home/DeepMar

3.执行步骤一训练流程（环境安装除外）

三、训练结果
/home/CRNN_for_Pytorch/test/output/0/