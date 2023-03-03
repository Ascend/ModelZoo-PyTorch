# EfficientNet-B3 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

EfficientNet-B3是一个新的卷积网络家族，与之前的模型相比，具有更快的训练速度和更好的参数效率。该模型通过一组固定的缩放系数统一缩放，这在网络深度，网络宽带，分辨率这三方面有明显优势。在 `EfficientNet` 中，这些特性是按更有原则的方式扩展的，也就是说，一切都是逐渐增加的。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/pycls
  commit_id=ee89cecb0e295b8037843e7e28344b156a847554
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

   用户自行获取原始数据集 `imagenet` ，将数据集上传到服务器并解压。

   数据集目录结构参考如下所示:
   ```
   ├──ImageNet
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

2. 数据预处理。
 
   在 `pycls/datasets/loader.py` 中修改数据集的路径，将变量 `_DATA_DIR` 修改为 `imagenet` 数据集的路径。


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
     bash ./test/train_eval_8p.sh --data_path=/data/xxx/ --pth_path=real_pre_train_model_path # 8卡评测
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。
   
   --pth_path参数填写训练权重生成路径，需写到权重文件的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --device                            //使用设备，gpu或npu
   --cfg                               //使用yaml配置文件路径
   --rank_id                           //默认卡号
   --device_id                         //默认设备号
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。
   

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V | - | - | 1 | - | 1.5 |
| 8p-竞品V | - | - | 100 | - | 1.5 |
| 1p-NPU |  -  | 356  | 1   |    O2    |  1.8  |
| 8p-NPU |  77.0613  | 2236  | 100   |    O2    |  1.8  |

# 版本说明
2022.08.01：更新pytorch1.8版本，重新发布。

## FAQ
无。






