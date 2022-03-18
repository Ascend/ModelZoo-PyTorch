# LSTM模型使用说明

## Requirements
* NPU配套的run包安装
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)
* (可选)参考《Pytorch 网络模型移植&训练指南》6.4.2章节，配置cpu为性能模式，以达到模型最佳性能；不开启不影响功能。

## 数据准备
1. 请搜索并下载TIMIT语音数据集
2. 将数据解压放置NPU目录下，数据集目录结构为
    ```
    LSTM
      |--NPU
          |--TIMIT
              |--DOC
              |--TEST
              |--TRAIN
    ```
       
## 安装依赖
    1. 安装kaldi(可选，首次处理TIMIT原始数据集需安装)
    
        chmod +x install_kaldi.sh
        ./install_kaldi.sh
    
    注意：install_kaldi.sh 根据所使用linux环境做适当修改。例如 centos 环境，将脚本中apt修改为yum;make -j 32, 数字32也可根据机器硬件条件相应修改
         请确认服务器环境网络通畅，否则会导致安装失败

    2. 安装依赖    
       
        pip3.7 install -r requirements.txt
    


## 训练模型

### 单卡训练流程：
    
    1. 进入NPU/1p 目录
    2. 修改字段device_id（单卡训练所使用的device id），为训练配置device_id，比如device_id=0
    3.a 首次训练需处理数据集，执行bash run_1p.sh （需安装kaldi, 即执行./install_kaldi.sh）
        该过程会生成处理好的数据放置于NPU/1p/data 目录下，后续训练或迁移至其他环境，可直接拷贝该文件到相同目录  
      b 非首次训练，将处理好的data数据拷贝到1p目录下，可减少数据处理时间开销
        执行 bash run_1p.sh 2
    

### 8卡训练流程
    
    1. 进入NPU/8p 目录
    2.a 首次训练需处理数据集，执行bash run_8p.sh （需安装kaldi, 即执行./install_kaldi.sh）
        该过程会生成处理好的数据放置于NPU/8p/data 目录下，后续训练或迁移至其他环境，可直接拷贝该文件到相同目录
      b 非首次训练，将处理好的data数据拷贝到8p目录下，可减少数据处理时间开销   
        执行 bash run_8p.sh 2
    

## Docker容器训练

1.导入镜像二进制包docker import ubuntuarmpytorch.tar REPOSITORY:TAG, 比如:

    docker import ubuntuarmpytorch.tar pytorch:b020


2.执行docker_start.sh后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：

    ./docker_start.sh pytorch:b020 /home/LSTM/NPU/data /home/LSTM


3.执行步骤一训练流程（环境安装除外）

## 训练结果

训练日志路径：在训练脚本的同目录下result文件夹里，如：

    ./result/training_1p_job_20201121023601
    ./result/training_8p_job_20201121023601


