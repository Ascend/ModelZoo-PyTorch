# DINO for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

DINO 是一个非监督的图像语义分割模型，模型结合transformer采用了一种没有标签的自蒸馏方法，
创造了一个teacher和一个student网络来处理非监督图像语义分割任务，
并基于小ViT模型产出的features，在k-NN分类器中达到78.1% top-1（Image-Net）。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/dino
  commit_id=cb711401860da580817918b9167ed73e3eef3dcf 
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                                           |
  |------------------------------------------------------------------------------| ------------------------------------------------------------ |
  | 硬件    | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)  |
  | 固件与驱动 | [5.1.rc2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/)                       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


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
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 性能训练
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/         # 精度训练
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 性能训练   
     ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                              //数据集路径
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type | CPU |
| :------: | :------:  | :------: | :------: | :------: |:------:|
| -        | 183      | 1        | 1        | O1       | ARM |
| -        | 1393      | 8        | 1        | O1       | ARM |
| -        | 190       | 1        | 1        | O1       | X86 |
| -        | 1329      | 8        | 1        | O1       | X86 |
| 69.7     | 1393      | 8        | 100      | O1       | ARM |

Note: ARM with 192 CPUs, X86 is Intel(R) Xeon(R) Platinum 8260 with 96 CPUs.


# 版本说明

## 变更

2022.12.23：Readme 整改。

## 已知问题

无。

