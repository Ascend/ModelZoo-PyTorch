# Swin-Transformer for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述
Swin-Transformer 使用层级式的transformer和移动窗口将transformer应用到CV领域。其通过限制在窗口内使用自注意力，带来了更高的效率 ，通过移动，使得相邻两个窗口之间有了交互，上下层之间也就有了跨窗口连接，从而变相达到了一种全局建模的效果。

- 参考实现：

  ```
  url=https://github.com/microsoft/Swin-Transformer
  commit_id=22e57f446ecc3fa650df1e1a271807bfd7ddcf74
  ```
- 适配昇腾 AI 处理器的实现：
  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
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

    **表 1** 版本配套表

       | 配套      | 版本                                                                               |
       |----------------------------------------------------------------------------------| ------------------------------------------------------------ |
       | 硬件  | [1.0.15.3](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)    |
       | NPU固件与驱动  | [22.0.0.3](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)    |
       | CANN    | [5.1.RC1.1](https://www.hiascend.com/software/cann/commercial?version=5.1.RC1.1) |
       | PyTorch | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)                           |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，下载开源数据[ImageNet](http://www.image-net.org/)训练集和验证集，将数据集上传到服务器任意路径下并解压。

2. 数据预处理。
  按照训练集格式处理验证集，可以使用以下[脚本](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)。
  以ImageNet数据集为例，数据集目录结构参考如下所示。

   ```
   ├── ImageNet
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
     bash ./test/train_full_1p.sh --data_path=real_data_path
     ```
     测试单卡性能。
     ```
     bash ./test/train_performance_1p.sh --data_path=real_data_path
     
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=real_data_path
     ```
      测试8卡性能。
     ```
     bash ./test/train_performance_8p.sh --data_path=real_data_path
     ```

   - 启动8卡评估。

     ```
     bash test/train_eval_8p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path
     ```



   --data_path参数填写数据集路径，--pth_path参数填写模型参数保存地址。

3. 模型训练脚本参数说明如下。

   ```
   --device_id             # 设备卡号
   --data_path             # 数据集路径
   --train_epochs          # 训练epoch
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示



**表 2** 训练结果展示表

| NAME      | Acc@1 |     FPS | Epochs | AMP_Type |
|-----------|-------|--------:|--------| -------: |
| 1p-竞品V    | -     |     284 | 1      |        - |
| 1p-竞品A    | -     |       - | 1      |        - |
| 1p-NPU1.8 | -     |  429.22 | 1      |       O2 |
| 8p-竞品V    | 81.1  |    1906 | 300    |        - |
| 8p-竞品A    | 81.1  |    2876 | 300    |        - |
| 8p-NPU1.8 | 81.0  | 3220.63 | 300    |       O2 |



# 版本说明

## 变更

2022.12.20：更新readme，重新发布。



## 已知问题



无。
