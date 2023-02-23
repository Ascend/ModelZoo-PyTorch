# Vit_small_patch16_224 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

Vit_small_patch16_224模型把Transformer设计思路用在视觉任务如图片分类上，通过图片分成一个个patch，然后把这些patch组合在一起作为对图像的序列化操作，就形成了类似文本类数据，从而扩展了视觉任务处理思路。

- 参考实现：

  ```
  url=https://github.com/rwightman/pytorch-image-models
  commit_id=a41de1f666f9187e70845bbcf5b092f40acaf097
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
  | PyTorch 1.5 | torchvision==0.2.2.post3|
  | PyTorch 1.8 | torchvision==0.9.1|

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  ```
  > **说明:** 只需执行一条对应的PyTorch版本依赖安装命令。
## 准备数据集

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

   --data_path参数填写数据集路径，需写到数据集的一级目录。


   模型训练脚本参数说明如下。

   ```
   公共参数：
   --model                              //使用模型，默认：vit_small_patch16_224
   -j                                   //加载数据进程数
   --opt                                //使用优化器,
   --epochs                             //重复训练次数
   -b                                   //训练批次大小
   --lr                                 //初始学习率
   --mixup                              //数据增强mixup系数
   --weight_decay                       //权重衰减
   --apex-amp                           //是否使用混合精度
   --drop                               //dropout系数
   --drop-path                          //drop-path系数
   --device_num                         //使用卡数量
   --npu                                //使用设备
   --combine_grad                       //使能combine_grad功能
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   Name  | Acc@1  |   FPS   | Epochs | AMP_Type |  Torch_Version |
| :----:  | :---:  |  :-----:| :----: | :------: |   :----: |
| 1p-竞品v |   -   | 586.67  |   1    |   O2    |   1.5   |
| 8p-竞品v | 67.65 | 304.06  |   1    |   O2    |   1.5   |
| 1p-Npu  |  -     | 4556.28 |  100   |   O2    |   1.8  |
| 8p-NPU  | 67.67  |  2373.80 | 100   |   O2    |   1.8  |

# 版本说明

## 变更

2020.10.14：更新内容，重新发布。

2020.07.08：首次发布。

## FAQ

无。