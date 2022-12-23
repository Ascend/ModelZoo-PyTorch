# CenterFace 训练
# the real-time face detection CenterFace
unofficial version of centerface, which achieves the best balance between speed and accuracy. CenterFace is a practical anchor-free face detection and alignment method for edge devices.
The project provides training scripts, training data sets, and pre-training models to facilitate users to reproduce the results. Finally, thank the centerface's author for the training advice.

## Requirements
use pytorch, you can use pip or conda to install the requirements
```
# for pip
cd $project
pip install -r requirements.txt

# for conda
conda env create -f enviroment.yaml
```

## 数据集准备
1. download the pretrained model from [Baidu](https://pan.baidu.com/s/1sU3pRBTFebbsMDac-1HsQA) password: etdi
2. download the validation set of [WIDER_FACE](https://pan.baidu.com/s/1b5Uku0Bb13Zk9mf7mkZ3FA) password: y4wg
3. the annotation file and train data can download for [Baidu](https://pan.baidu.com/s/1j_2wggZ3bvCuOAfZvjWqTg) password: f9hh

1)本机解压WIDER_FACE_DATA_ALL.zip文件里面有annotations.zip、labels、WIDER_train.zip、WIDER_val.zip、groud_truth文件。
2）annotations.zip、labels、WIDER_train.zip、WIDER_val.zip复制到服务器的$project/data/wider_face目录下。groud_truth复制到$project下。
3) 将WIDER_train中的images,复制到$project/data/wider_face/image

## Training

一、训练流程

单卡训练流程：

    1.安装环境
    2.编译（编译过的可跳过，编译需要先执行以下操作，否找可能出现报错ModuleNotFoundError: No module named 'external.nms'）
        cd $project/src/lib/external
        make    
    3.开始训练
        bash ./test/train_full_1p.sh  --data_path=数据集路径         # 精度训练
        bash ./test/train_performance_1p.sh  --data_path=数据集路径  # 性能训练


​    
多卡训练流程

    1.安装环境
    2.编译（编译过的可跳过，编译需要先执行以下操作，否找可能出现报错ModuleNotFoundError: No module named 'external.nms'）
        cd $project/src/lib/external
        make 
    3.开始训练
        bash ./test/train_full_8p.sh  --data_path=数据集路径         # 精度训练
        bash ./test/train_performance_8p.sh  --data_path=数据集路径  # 性能训练


二、Docker容器训练

1.导入镜像二进制包docker import ubuntuarmpytorch.tar REPOSITORY:TAG, 比如:

    docker import ubuntuarmpytorch.tar pytorch:b020

2.执行docker_start.sh后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：


    ./docker_start.sh pytorch:b020 /train/peta /home/DeepMar

3.执行步骤一训练流程（环境安装除外）

三、测试结果
训练日志路径：网络脚本test下output文件夹内。例如：
      test/output/devie_id/CenterFace_${device_id}.log          # 训练脚本原生日志
      test/output/devie_id/CenterFace_bs1024_8p_perf.log  # 8p性能训练结果日志
      test/output/devie_id/CenterFace_bs1024_8p_acc.log   # 8p精度训练结果日志

训练模型：训练生成的模型默认会写入到和test文件同一目录下。当训练正常结束时，checkpoint.pth.tar为最终结果。

### CenterFace training result

| ***\*测试项\**** | ***\*超参信息\****                    | ***\*NPU\****                     | ***\*测试结果\**** |
| ---------------- | ------------------------------------- | --------------------------------- | ------------------ |
| Train-1p:性能    | batch_size=32 lr=5e-4 lr_step='75,95' | 34.5                              | OK                 |
| Train-1p:精度    |                                       |                                   |                    |
| Train-8p:性能    | batch_size=16 lr=2.5e-3 epochs=140    | 41.5                              | /                  |
| Train-8p:精度    |                                       | Easy:87.03;medium:86.50hard:70.17 | OK                 |