# Transformer-SSL for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

Transformer-SSL 是一个以Swin-Transformer作为骨干网络的自监督模型，该模型通过将 MoCov2 和 BYOL 相结合，得到了MoBY模型。


- 参考实现：

  ```
  url=https://github.com/SwinTransformer/Transformer-SSL
  commit_id=4510de1f21ee6f9810a74494c254081dd8f2c383
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.6.0；pillow==8.4.0 |
  | PyTorch 1.8 | torchvision==0.9.1；pillow==9.1.0 |
  | PyTorch 1.11 | timm==0.4.5 |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  
  pip install -r 1.11_requirements.txt  # PyTorch1.11版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。

## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，将数据集上传到服务器并解压。

   数据集目录结构参考如下所示:
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

   
   ```
   bash ./test/train_performance_1p.sh --data_path=数据集路径  # 1p性能
   ```
   
   - 单机8卡训练
   
   
   ```
   bash ./test/train_performance_8p.sh --data_path=数据集路径  # 8p性能
   bash ./test/train_full_8p.sh --data_path=数据集路径  # 8p精度
   ```

   **脚本默认情况下会将 checkpoint 保存在 `output` 目录下。 如果想要从头开始训练，请先移除`output`目录**

   --data_path参数填写数据集路径，需写到数据集的一级目录。
   
   模型训练脚本参数说明如下。
   
   ```
   公共参数： 
   --batch-size                        //训练批次大小
   --amp-opt-level                     //使用混合精度类别
   --max_epochs                        //epoch数目，默认为100
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|  NAME   | Acc@1 |  FPS   | Epochs   | AMP_Type | Torch_Version | CPU |
|:-------:|:-----:|:------:| :------: | :------: |:---:|-----|
| 1p_NPU |   -   |  171   | 1        | O1       | 1.5 | X86 |
| 8p_NPU | 67.48 |  1256  | 100      | O1       | 1.5 | X86 |
| 1p_NPU |   -   | 161.39 | 1        | O1       | 1.8 | X86 |
| 8p_NPU | 68.35 |  1199  | 100      | O1       | 1.8 | X86 |
| 1p_NPU | -        | 140       | 1        | O1       | 1.8 | ARM |
| 8p_NPU | 67.48    | 1150      | 100      | O1       | 1.8 | ARM |
| 8p_NPU | 74.14 |  1150  | 300      | O1       | 1.8 | ARM |
| 16p_NPU | 68.47 |  1845  | 100      | O1       | 1.8 | ARM |


# 版本说明
## 变更

2023.01.09：整改readme，重新发布。

## FAQ
无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md