# Convmixer for PyTorch 

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述 

## 简述

Convmixer由一个patch embedding层和一个简单的全卷积块的重复应用组成，并保持着patch embedding的空间结构。Convmixer是一个非常简单的卷积架构，直接在patch上操作，它在所有层中保持相同分辨率和大小的表示。Convmixer证明了patch表示本身可能会是Vision transformer这样的新架构卓越性能的最关键组件。

- 参考实现：
  ```
  url=https://github.com/locuslab/convmixer.git
  commit_id=47048118e95721a00385bfe3122519f4b583b26e
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
  | PyTorch 1.5 | torchvision==0.6.0；pillow==8.4.0 |
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
1. 获取数据集 

     请用户自行下载**ImageNet**数据集，将数据集上传到服务器任意路径下并解压。
     数据集目录结构参考如下所示。
     
    ```
    ├── imagenet
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

   该模型支持单机单卡性能和单机8卡训练。
    - 单机单卡性能
   
      启动单卡性能测试。

      ```
      bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
      ```

    - 单机8卡性能
   
      启动8卡性能测试。

      ```
      bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度

      bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
      ```

    - 单机单卡评测
   
      启动单卡评测。
   
      ```
      bash ./test/train_eval_1p.sh --data_path=/data/xxx/ --checkpoint=ckpt_path
      ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。
   
   --checkpoint参数填写训练生成的权重文件路径，需写到文件的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --weight-decay                      //权重衰减
   --batch-size                        //训练批次大小
   --input-size                        //输入图片大小
   --epochs                            //重复训练次数
   --lr                                //初始学习率，默认：0.01
   --num-classes                       //分类数
   --amp                               //是否使用混合精度
   --device                            //指定训练设备
   --seed                              //随机数种子设置
   --momentum                          //动量
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|  NAME   | Acc@1   |   FPS   |   Epochs   | AMP_Type | Torch_Version |
|:-------:| :-----: |:-------:|:----------:| :------: | :-----: |
| 1p-NPU  |   -     |  42.09  |     1      |    O2    |   1.8   |
| 8p-NPU  | 80.2%   | 376.64  |    150     |    O2    |   1.8   |

# 版本说明

## 变更

2023.02.24：更新内容，重新发布。

2020.07.08：首次发布。

### FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
