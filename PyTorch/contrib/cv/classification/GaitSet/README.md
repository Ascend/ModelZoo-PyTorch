# GaitSet for PyTorch

- [概述](概述.md)
- [准备训练环境](准备训练环境.md)
- [开始训练](开始训练.md)
- [训练结果展示](训练结果展示.md)
- [版本说明](版本说明.md)

# 概述

## 简述

GaitSet是一个灵活、有效和快速的跨视角步态识别网络。灵活性：其输入可以是轮廓组成的集合，并没有其他约束。有效性：它在CASIA-B数据集上达到95.0%的准确率。快速性：使用8个NVIDIA 1080TI GPU，它只需7分钟即可在OU-MVLP数据集上完成评估，这个数据集包含13w序列，平均每条序列有70个图像。

- 参考实现：

  ```
  url=https://github.com/AbnerHqC/GaitSet
  commit_id=5535943428b66415530d8379b648b8f74a294219
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

  **表 1**  版本配套表

  | 配套       | 版本                                                        |
  | ---------- | ------------------------------------------------------------ |
  | 硬件 | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | NPU固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/) |

- 环境准备指导。

  请参考《 [Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes) 》。
  
 - 安装依赖。
    ```
    pip install -r requirements.txt
    ```
    **表 2** 依赖列表

    | 依赖       | 版本         |
    | ---------- |------------|
    |numpy| 1.20.1     |
    |opencv-python| 4.5.2.54   |
    |imageio| 2.9.0      |
    |xarray| 0.18.2     |
    | apex    | 0.1+ascend |
  

## 准备数据集

1. 获取数据集
    
    下载`CASIA-B`数据集：
    
    > 数据集主页：http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp
    >
    > 数据集下载地址：http://www.cbsr.ia.ac.cn/GaitDatasetB-silh.zip

   用户根据数据集下载地址下载数据集，将数据集上传到服务器任意路径下并解压，解压后数据集内部的目录应为（`CASIA-B`数据集）：数据集路径/对象序号/行走状态/角度，例如`CASIA-B/001/nm-01/000/ `。

   `CASIA-B`数据集目录结构参考如下所示。

   ```
   ├── CASIA-B
         ├──001
              ├──bg-01
                    │──000
                        │──图片1
                        │──图片2
                        │   ...  
                    │──018
                        │──图片1
                        │──图片2
                        │   ... 
                    │   ...       
              ├──cl-01
                     │──000
                        │──图片1
                        │──图片2
                        │   ...  
                     │──018
                        │──图片1
                        │──图片2
                        │   ... 
                     │   ...    
              ├──nm-01
                     │──000
                        │──图片1
                        │──图片2
                        │   ...  
                     │──018
                        │──图片1
                        │──图片2
                        │   ... 
                     │   ...    
              ├──...                     
         ├──002  
              ├──bg-01
                    │──000
                        │──图片1
                        │──图片2
                        │   ...  
                    │──018
                        │──图片1
                        │──图片2
                        │   ... 
                    │   ...       
              ├──cl-01
                     │──000
                        │──图片1
                        │──图片2
                        │   ...  
                     │──018
                        │──图片1
                        │──图片2
                        │   ... 
                     │   ...    
              ├──nm-01
                     │──000
                        │──图片1
                        │──图片2
                        │   ...  
                     │──018
                        │──图片1
                        │──图片2
                        │   ... 
                     │   ...    
              ├──... 
         ├──...     
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

3. 数据预处理。

    使用`pretreatment.py`进行数据处理：其中，包括括号"{}"需要替换为数据集的路径
    
    ```bash
    # --input_path为原数据集‘CASIA-B’的路径； --output_path为预处理后的数据集路径。
    $ python3 pretreatment.py --input_path {downloaded_path} \
                              --output_path {output_path}
    ```

    >  预处理过程中提示`--WARNING--`属于预期现象，请等待处理完成

# 开始训练

## 训练模型


1. 进入解压后的源码包根目录。

    ```bash
    $ cd /${模型文件夹名称} 
    ```

2. 运行训练脚本。

    该模型支持单机单卡训练和单机8卡训练。
- 单机单卡训练

    NPU1P训练

    ```bash
   # --data_path参数填写数据集预处理后所在路径。
    $ bash test/train_full_1p.sh --data_path=${data_path}
    ```
   
    训练 1p 性能
    ```bash
   # --data_path参数填写数据集预处理后所在路径。
    $ bash test/train_performance_1p.sh --data_path=${data_path}
    ```

    RT训练脚本可以外部指定数据集路径--data_path和迭代数--iters

    RT1脚本1p训练
    ```bash
    $ bash test/train_ID4118_GaitSet_RT1_performance_1p.sh --data_path=${data_path} --iters=${iters}
    ```
    
    RT2脚本1p训练
    ```bash
    $ bash test/train_ID4118_GaitSet_RT2_performance_1p.sh --data_path=${data_path} --iters=${iters}
    ```
- 单机8卡训练

    NPU8P训练

    ```bash
   # --data_path参数填写数据集预处理后所在路径。
    $ bash test/train_full_8p.sh --data_path=${data_path}
    ```
    
    训练 8p 性能
    ```bash
   # --data_path参数填写数据集预处理后所在路径。
    $ bash test/train_performance_8p.sh --data_path=${data_path}
    ```
- 模型验证

    ```bash
  # --data_path参数填写数据集预处理后所在路径。
    $ bash test/train_eval_8p.sh --data_path=${data_path}
    ```
    注意：需要手动把`train_eval_8p.sh`中`--iter`参数改为训练保存的模型想要加载的代数。

# 训练结果展示
**表 3** 训练结果展示表

|   | 训练代数(Iters)  |  精度(RANK-1, %) |  性能(FPS) | PyTorch版本 |
|---|---|---|-----------|---|
|  NPU1P |  6w |  95.488 | 270  | 1.5       |
| NPU8P|  4w |  95.525 | 1000  | 1.5       |
|  NPU1P |  6w |   |  270.696 | 1.8       |
|  NPU8P |  4w |  93.744 | 1222.591  | 1.8       |

# 版本说明

## 变更

2022.11.26：更新内容，重新发布。

## 已知问题

无。











