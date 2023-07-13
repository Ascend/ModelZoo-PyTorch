# Evo-Levit for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

Evo-ViT的具体框架设计，包括基于全局class attention的token选择以及慢速、快速双流token更新两个模块。其根据全局class attention的排序判断高信息token和低信息token，将低信息token整合为一个归纳token，和高信息token一起输入到原始多头注意力（Multi-head Self-Attention, MSA）模块以及前向传播（Fast Fed-forward Network, FFN）模块中进行精细更新。更新后的归纳token用来快速更新低信息token。全局class attention也在精细更新过程中进行同步更新变化。

- 参考实现：

  ```
  url=https://github.com/YifanXu74/Evo-ViT
  commit_id=4c5d9b30b0a3c9b1e7b8687a9490555bd9d714ca
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
  
- 关于 `timm` 包的NPU优化补丁。

  ```
  # 需要先cd到模型源码包根目录下
  # 执行以下命令，先后生成补丁并升级包
  diff -uN {timm_path}/data/mixup.py ./fix_timm/mixup.py >mixup.patch
  diff -uN {timm_path}/optim/optim_factory.py ./fix_timm/optim_factory.py >optim.patch
  patch -p0 {timm_path}/data/mixup.py mixup.patch
  patch -p0 {timm_path}/optim/optim_factory.py optim.patch
  ```
  > **说明：**
  > timm_path为timm包的安装路径，一般timm包的安装位置在/usr/local/lib/python3.7/dist-packages/timm/。

  


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集 `ImageNet2012` ，将数据集上传到服务器任意路径下并解压。

   以 `ImageNet2012` 数据集为例，数据集目录结构参考如下所示。

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

## 获取预训练模型

`Evo-Vit` 模型训练需要配置 `teacher—model` ，用户自行获取 `regnety_160-a5fe301d.pth` 预训练模型，可参考GitHub的[Evo-Vit](https://github.com/YifanXu74/Evo-ViT)。将获取的预训练模型放置在源码包根目录下。与源码中的配置参数的默认值 `./regnety_160-a5fe301d.pth` 保持一致。

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
     bash ./test/train_full_1P.sh --data_path=/data/xxx/  # 单卡精度
     
     bash ./test/train_performance_1P.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8P.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8P.sh --data_path=/data/xxx/  # 8卡性能
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                         //数据集路径     
   --model                             //模型名称
   --batch-size                        //训练批次大小
   --input-size                        //输入图像大小
   --output_dir                        //输出路径
   ```
   
   训练完成后，权重文件保存在当前路径的save中，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V | - | 51 | 1 | O1 | 1.8 |
| 8p-竞品V | 73.54 | 487 | 100 | O1 | 1.8 |
| 1p-NPU | - | 66.93 | 1 | O1 | 1.8 |
| 8p-NPU | 74.32 | 510.72 | 100 | O1 | 1.8 |

# 版本说明

## 变更

2022.11.09：首次发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md