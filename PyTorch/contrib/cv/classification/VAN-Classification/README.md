# VAN-Classification for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述

自注意力机制将2D图像视为1D序列，这会破坏图像的关键2D结构。由于其二次计算和内存开销，处理高分辨率图像也很困难。此外，自注意力机制是一种特殊的注意，它只考虑空间维度的适应性，而忽略了通道维度的适应性，这对视觉任务也很重要。 VAN使用了一种新的计算机视觉注意机制LKA，它既考虑了卷积和自我注意的优点，又避免了它们的缺点。

- 参考实现：

  ```
  url=https://github.com/Visual-Attention-Network/VAN-Classification
  commit_id=e19779b53a1b0828b51ecb4412d577541aee83a7
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
  | PyTorch 1.5 | torchvision==0.2.2.post3 |
  | PyTorch 1.8 | torchvision==0.9.1 |
  
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
   > 该数据集的训练过程脚本只作为一种参考示例。

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
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh ${data_path} ${ckpt_pth}
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   ckpt_pth参数填写训练生成的权重文件路径，需写到文件的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --workers                           //加载数据进程数   
   --model                             //要训练的模型
   --img-size                          //图像patch大小
   --opt                               //优化器，默认adamw
   --momentum                          //动量，默认0.9
   --weight-decay                      //权重衰减，默认0.05
   --lr                                //学习率，默认1e-3
   --epochs                            //重复训练次数，默认300
   --amp                               //是否使用混合精度
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V |   -   | 137  |   1    |    O1    |      1.5      |
| 8p-竞品V |   -   | 1296 |  310   |    O1    |      1.5      |
|  1p-NPU  | 82.6  | 297  |   1    |    O1    |      1.5      |
|  8p-NPU  | 82.4  | 2184 |  310   |    O1    |      1.5      |

# 版本说明

## 变更

2023.02.24：更新readme，重新发布。

2022.09.01：首次发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md