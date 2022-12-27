# UNet for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述
UNet在生物医学图像分割领域，得到了广泛的应用。
它是完全对称的，且对解码器（应该自Hinton提出编码器、解码器的概念来，
即将图像->高语义feature map的过程看成编码器，高语义->像素级别的分类score map的过程看作解码器）
进行了加卷积加深处理。

- 参考实现：

  ```
  url=https://github.com/4uiiurz1/pytorch-nested-unet
  commit_id=557ea02f0b5d45ec171aae2282d2cd21562a633e
  ```
- 适配昇腾 AI 处理器的实现：
  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/semantic_segmentation
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

     | 配套      | 版本                                                                          |
     |-----------------------------------------------------------------------------| ------------------------------------------------------------ |
     | 硬件  | [1.0.13](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
     | NPU固件与驱动  | [21.0.4](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
     | CANN    | [5.0.4](https://www.hiascend.com/software/cann/commercial?version=5.0.4)    |
     | PyTorch | [1.5](https://gitee.com/ascend/pytorch/tree/v1.5.0/)                        |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。 从 [data-science-bowl-2018](https://www.kaggle.com/c/data-science-bowl-2018/data)获取数据集。
2. 上传数据集在源码包根目录下新建的`inputs`文件夹下并解压。
3. 数据集需要执行预处理，在源码包根目录下执行
    ```bash
    python3.7 preprocess_dsb2018.py
    ```
4. 数据集目录结构参考如下所示。
    ```
    inputs
    └── data-science-bowl-2018
        ├── stage1_train
        |   ├── 00ae65...
        │   │   ├── images
        │   │   │   └── 00ae65...
        │   │   └── masks
        │   │       └── 00ae65...            
        │   ├── ...
        |
    ...
    ```



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
     bash ./test/train_full_1p.sh --data_path={data/path} # train accuracy
     ```
     测试单卡性能。
     ```
     bash ./test/train_performance_1p.sh --data_path={data/path} # train performance
     
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path={data/path} # train accuracy
     ```
      测试8卡性能。
     ```
     bash ./test/train_performance_8p.sh --data_path={data/path} # train performance
     ```

   - 启动8卡评估。

     ```
     bash ./test/train_eval_8p.sh --data_path={data/path}
     ```

   - onnx转换。

     ```
     python3.7.5 pthtar2onnx.py
     ```
     
   - 在线推理示例。
     ```
     python3.7.5 demo.py (分割后的图片存储在outputs/UNet_Demo/)
     ```

   --data_path参数填写数据集路径。

3. 模型训练脚本参数说明如下。

   ```
   --device_id             # 设备卡号
   --data_path             # 数据集路径
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示



**表 2** 训练结果展示表


| NAME   | 精度    |  FPS | Epochs | AMP_Type |
|--------|-------|-----:|--------| -------: |
| 1p-GPU | -     |  823 | 1      |        - |
| 1p-NPU | -     |  238 | 1      |       O2 |
| 8p-GPU | 83.91 | 2332 | 100    |        - |
| 8p-NPU | 83.31 |  723 | 100    |       O2 |


# 版本说明

## 变更

2022.12.20：整改readme，重新发布。



## 已知问题

无。
