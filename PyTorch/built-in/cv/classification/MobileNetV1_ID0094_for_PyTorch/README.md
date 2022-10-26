# MobileNetv1
Implementation of MobileNet, modified from https://github.com/wjc852456/pytorch-mobilenet-v1

imagenet data is processed [as described here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset)

## 一、依赖

    NPU配套的run包安装
    Python 3.7.5
    PyTorch(NPU版本)
    apex(NPU版本)
    torch(NPU版本)
    torchvision
    pillow
    Installation guidance: https://gitee.com/ascend/pytorch/blob/master/docs/zh/PyTorch%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97/PyTorch%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97.md 

## 二、训练流程：

注意：data_path为数据集imagenet所在的路径；
     pillow建议安装较新版本，与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision，建议Pillow版本是9.1.0 torchvision版本是0.6.0

- 环境变量配置

```shell
训练前执行以下命令：
source /usr/local/Ascend/ascend-toolkit/set_env.sh	
source env_npu.sh					
```
- 单卡训练流程：

```shell
1. 安装环境
2. 开始训练
    bash test/train_full_1p.sh  --data_path=/data/imagenet
```
- 多卡训练流程

```shell
1. 安装环境
2. 开始训练
    bash test/train_full_8p.sh  --data_path=/data/imagenet
```

## 三、测试结果

- 训练日志路径：在训练脚本的同目录下result文件夹里，如：
```
    /home/MobileNet/test/output/
```    
- 结果
```
paper:                  top1 70.6

pytorch 1.8 (using sgd optimizer)
| Npu_nums | Epoch     | Acc@1    | Acc@5    | FPS      | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: | :------: |
| 1        | 1         | 23.170   | 47.908   | 2774.9   | O2       |
| 8        | 90(final) | 69.813   | 89.417   | 10465.7  | O2       |
| 8        | 89(best)  | 70.023   | 89.552   | 10465.7  | O2       |

pytorch 1.5 Results by former contributor
- sgd :                    top1 68.848 top5 88.740
- rmsprop:                top1 0.104  top5 0.494
- rmsprop init from sgd :  top1 69.526 top5 88.978
```