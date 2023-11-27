# SPACH for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)


# 概述

## 简述

SPACH是由来自Microsoft Research和中国科学技术大学的学者提出的神经网络框架，在论文中将其作为统一框架对Transformer，MLP，CNN模型进行了对比。 这个框架总体来说有两种模式：多阶段和单阶段。每个阶段内部采用的是Mixing Block，而该Mixing Block可以是卷积层、Transformer层以及MLP层。本文使用PyTorch实现了使用SPACH训练imagenet数据集的具体实例。

- 参考实现：

    ```
    url=https://github.com/microsoft/SPACH.git
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

- 安装numactl。

  ```
  apt-get install numactl  # for Ubuntu
  yum install numactl  # for CentOS
  ```


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
     bash ./test/train_eval_8p.sh --data_path=/data/xxx --resume=ckpt_path
     ```
   

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   --resume参数填写模型训练生成的ckpt文件路径，写到文件的一级目录即可。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data-path                         //数据集路径
   --model                             //使用模型  
   --num_workers                       //加载数据进程数
   --weight-decay                      //权重衰减
   --npu                               //是否使用npu
   --batch-size                        //训练批次大小
   --dist-eval                         //启用分布式评估
   --epochs                            //训练周期数
   --seed                              //随机数种子设置
   ```
   
    训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 |  FPS | Epochs | AMP_Type | Torch_Version |
| :-----: |:---: | :--: | :----: | :-----: | :------: |
| 1p-竞品V | -   |298.9424 | 1      |       O2 | 1.5 |
| 8p-竞品V | 81.6 |2604.2726 | 300 |    O2    | 1.5 |
| 1p-NPU  | -   |384.99| 1      |       O2 | 1.8 |
| 8p-NPU  | 81.23 |2936.68 | 300    |       O2 | 1.8 |



# 版本说明

## 变更

2023.02.23：更新readme，重新发布。

2022.04.27：首次发布。

## FAQ
无。


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md