# SK-resNet50

## ImageNet training with PyTorch

This implements training of SK-resNet50 on the ImageNet dataset, mainly modified from [Github](https://github.com/pytorch/examples/tree/master/imagenet).

## SK-resNet50 Detail

Base version of the model from [pytorch.torchvision](https://github.com/implus/PytorchInsight/blob/master/classification/models/imagenet/resnet_sk.py).
The training script is adapted from [training script on imagenet](https://github.com/implus/PytorchInsight/blob/master/classification/imagenet_fast.py).

## Requirements

- pytorch_ascend, apex_ascend, tochvision
  Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).

## Training

一、训练流程

单卡训练流程：

    1.安装环境
    2.修改参数:
       device_list（训练时可见的 device id），建议只配置训练使用的 device_id，比如 --device_list '0'   
    3.开始训练
        bash ./test/train_full_1p.sh  --data_path=数据集路径         # 精度训练
        bash ./test/train_performance_1p.sh  --data_path=数据集路径  # 性能训练


​    
多卡训练流程

    1.安装环境
    2.开始训练
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
      test/output/devie_id/train_${device_id}.log          # 训练脚本原生日志
      test/output/devie_id/SkresNet50_bs1024_8p_perf.log  # 8p性能训练结果日志
      test/output/devie_id/SkresNet50_bs1024_8p_acc.log   # 8p精度训练结果日志

训练模型：训练生成的模型默认会写入到和test文件同一目录下。当训练正常结束时，checkpoint.pth.tar为最终结果。

### SK-resNet50 training result

| Acc@1  | FPS  | Npu_nums | Epochs | Type |
| :----: | :--: | :------- | :----: | :--: |
|   -    | 820  | 1        |   1    |  O2  |
| 76.838 | 6400 | 8        |  100   |  O2  |

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
