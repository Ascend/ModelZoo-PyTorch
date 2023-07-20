# Resnext50_32x4d for PyTorch

- [概述](https://gitee.com/ascend/docs-openmind/blob/master/guide/modelzoo/pytorch_model/tutorials/%E6%A6%82%E8%BF%B0.md)
- [准备训练环境](https://gitee.com/ascend/docs-openmind/blob/master/guide/modelzoo/pytorch_model/tutorials/%E5%87%86%E5%A4%87%E8%AE%AD%E7%BB%83%E7%8E%AF%E5%A2%83.md)
- [开始训练](https://gitee.com/ascend/docs-openmind/blob/master/guide/modelzoo/pytorch_model/tutorials/%E5%BC%80%E5%A7%8B%E8%AE%AD%E7%BB%83.md)
- [训练结果展示](https://gitee.com/ascend/docs-openmind/blob/master/guide/modelzoo/pytorch_model/tutorials/%E8%AE%AD%E7%BB%83%E7%BB%93%E6%9E%9C%E5%B1%95%E7%A4%BA.md)
- [版本说明](https://gitee.com/ascend/docs-openmind/blob/master/guide/modelzoo/pytorch_model/tutorials/%E7%89%88%E6%9C%AC%E8%AF%B4%E6%98%8E.md)

# 概述

## 简述

ResNeXt-50-32x4d是一个经典的图像分类网络，对于一个L层的网络，相较于ResNet网络，ResNeXt-50-32x4d采用了组卷积的方法，在分组进行卷积后再进行concat拼接，可以大大减少网络的参数量，对于卷积核大小相同的两个网络，普通卷积的参数量会远远大于组卷积的参数量。这些特点让ResNeXt-50-32x4d在参数和计算成本更少的情形下实现比ResNet更优的性能。

- 参考实现：
  
  ```
  url=https://github.com/pytorch/examples/tree/master/imagenet
  commit_id=49e1a8847c8c4d8d3c576479cb2fe2fd2ac583de
  ```
  
- 适配昇腾 AI 处理器的实现：
  
  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.2.2.post3；pillow==8.4.0 |
  | PyTorch 1.8 | torchvision==0.9.1；pillow==9.1.0 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
    
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。

## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，CIFAR-10等，将数据集上传到服务器任意路径下并解压。

   以ImageNet2012数据集为例，数据集目录结构参考如下所示。

    ```
    ├──ImageNet2012
        ├──train
             ├──类别1
                   │──图片1
                   │──图片2
                   │   ...       
             ├──类别2
                   │──图片1
                   │──图片2
                   │   ...   
             ├──...                     
        ├──val  
             ├──类别1
                   │──图片1
                   │──图片2
                   │   ...       
             ├──类别2
                   │──图片1
                   │──图片2
                   │   ...                         
    ```
    > **说明：** 该数据集的训练过程脚本只作为一种参考示例。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

  - 单机单卡训练
    
    启动单卡训练。
    
    ```
    bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
    
    bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
    ```
    
  - 单机8卡训练
    
    启动8卡训练。
    
    ```
    bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
    
    bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
    ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh --data_path=/data/xxx/  # 对应修改脚本中resume参数，指定为训练生成的ckpt文件路径
     ```

  --data_path参数填写数据集路径，需写到数据集的一级目录。

  模型训练脚本参数说明如下。

  ```
  公共参数：
  --addr                              //主机地址
  --arch                              //使用模型，默认：resnet34
  --workers                           //加载数据进程数  
  --epochs                            //重复训练次数
  --batch-size                        //训练批次大小
  --lr                                //初始学习率，默认：0.1
  --momentum                          //动量，默认：0.9
  --amp                               //是否使用混合精度
  --loss-scale                        //混合精度lossscale大小
  --opt-level                         //混合精度类型
  多卡训练参数：
  --multiprocessing-distributed       //是否使用多卡训练
  --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
  ```

  训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2** 训练结果展示表

| NAME | Acc@1 | FPS | Epochs | AMP_Type | Torch_Version |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 1p-NPU | -   | 953.08 | 1 | O2 | 1.8 |
| 8p-NPU | 77.22 | 7239.146 | 90 | O2 | 1.8 |

# 版本说明

## 变更

2023.02.16：更新readme，重新发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md