# TResNet for PyTorch


-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)


# 概述

## 简述

TResNet的设计基于ResNet50架构，并进行专门的修改和优化，旨在不失训练和推理速度的前提下，提升网络的精度，其优化的方面为SpaceToDepth stem、抗锯齿降采样、原地激活批归一化、Block类型选择方法、优化的SE层。 TResNet 在 Top-1 准确度上超越了 ResNet50。

- 参考实现：

    ```
    url=https://github.com/rwightman/pytorch-image-models/tree/v0.4.5 
    commit_id=5b28ef410062af2a2ab1f27bf02cf33e2ba28ca2
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

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，将数据集上传到服务器任意路径下并解压。

   以ImageNet2012数据集为例，数据集目录结构参考如下所示。

   ```
   ├── ImageNet2012
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

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

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
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 单卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_1p.sh --data_path=/data/xxx/ --pth_path=real_pre_train_model_path
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。
   
   --pth_path参数填写训练权重生成路径，需写到权重文件的一级目录。
   
   模型训练脚本参数说明如下。

   ```
   公共参数：
   --model                             //使用模型  
   --workers                           //加载数据进程数
   --lr                                //初始学习率
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小
   --world-size                        //分布式训练节点数量
   --amp                               //是否使用混合精度
   --weight-decay                      //权重衰减
   --loss-scaler                       //混合精度loss scale大小
   ```
   
    训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 |  FPS | Epochs | AMP_Type | Torch_version |
| :-----: |:---: | :--: | :----: | :-----: | :------:|
| 1p-竞品V | -     |751.26 | 1      |       O2 | 1.5 |
| 8p-竞品V | 78.88 |5136.17 | 110 |       O2 | 1.5 |
| 1p-NPU  | -     |721.21| 1      |       O2 | 1.8 |
| 8p-NPU  |78.7219|5517.23 | 100  |       O2 | 1.8 |

# 版本说明

## 变更

2023.02.23：更新readme，重新发布。

2021.09.10：首次发布。


## FAQ

无。
