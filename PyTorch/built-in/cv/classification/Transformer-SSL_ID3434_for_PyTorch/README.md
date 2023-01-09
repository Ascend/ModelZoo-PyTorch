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

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 硬件       | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动  | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |


- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。
  ```
  pip install timm==0.3.2 --no-dependencies
  pip install -r requirements.txt
  ```


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
   bash ./test/train_full_8p.sh --data_path=数据集路径         # 8p精度
   ```

   **脚本默认情况下会将 checkpoint 保存在 `output` 目录下。 如果想要从头开始训练，请先移除`output`目录**

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

|  NAME   | Acc@1 |  FPS   | Npu_nums | Epochs   | AMP_Type | CPU |
|:-------:|:-----:|:------:| :------: | :------: | :------: |:---:|
| NPU+1.5 |   -   |  171   | 1        | 1        | O1       | X86 |
| NPU+1.5 | 67.48 |  1256  | 8        | 100      | O1       | X86 |
| NPU+1.8 |   -   | 161.39 | 1        | 1        | O1       | X86 |
| NPU+1.8 | 68.35 |  1199  | 8        | 100      | O1       | X86 |
| NPU+1.8 | -        | 140       | 1        | 1        | O1       | ARM |
| NPU+1.8 | 67.48    | 1150      | 8        | 100      | O1       | ARM |
| NPU+1.8 | 74.14 |  1150  | 8        | 300      | O1       | ARM |
| NPU+1.8 | 68.47 |  1845  | 16       | 100      | O1       | ARM |


# 版本说明
## 变更

2023.01.09：整改Readme发布。

## 已知问题
无。
