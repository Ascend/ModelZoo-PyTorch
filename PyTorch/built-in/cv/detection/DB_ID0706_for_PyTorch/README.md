# DBNet for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)



# 概述

## 简述

DB(Differentiable Binarization)是一种使用可微分二值图来实时文字检测的方法，
和之前方法的不同主要是不再使用硬阈值去得到二值图，而是用软阈值得到一个近似二值图，
并且这个软阈值采用sigmod函数，使阈值图和近似二值图都变得可学习。

- 参考实现：

  ```
  url=https://github.com/MhLiao/DB
  commit_id=4ac194d0357fd102ac871e37986cb8027ecf094e
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/detection
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | pillow==8.4.0 |
  | PyTorch 1.8 | pillow==9.1.0 |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装geos，可按照环境选择以下方式：

  1. ubuntu系统：

     ```
     sudo apt-get install libgeos-dev
     ```

  2. euler系统：

     ```
     sudo yum install geos-devel
     ```

  3. 源码安装：

     ```
     wget http://download.osgeo.org/geos/geos-3.8.1.tar.bz2
     bunzip2 geos-3.8.1.tar.bz2
     tar xvf geos-3.8.1.tar
     cd geos-3.8.1
     ./configure && make && make install
     ```

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
    
    请用户自行下载 `icdar2015` 数据集，解压放在任意文件夹 `datasets`下，数据集目录结构参考如下所示。

    ```
    |--datasets
       |--icdar2015
    ```

    > **说明：** 
    >该数据集的训练过程脚本只作为一种参考示例。

## 获取预训练模型

请用户自行获取预训练模型，将获取的 `MLT-Pretrain-Resnet50` 预训练模型，放至在源码包根目录下新建的 `path-to-model-directory` 目录下。


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
      1.安装环境，确认预训练模型放置路径，若该路径路径与model_path默认值相同，可不传参，否则执行训练脚本时必须传入model_path参数；
      2.开始训练
        bash ./test/train_full_1p.sh --data_path=${datasets} --model_path=${pretrain_model}    # 单卡精度
        bash ./test/train_performance_1p.sh --data_path=${datasets} --model_path=${pretrain_model}    # 单卡性能   
      ```
      **注意**：如果发现打屏日志有报checkpoint not found的warning，请再次检查预训练模型MLT-Pretrain-Resnet50的配置，以免影响精度。

   - 单机8卡训练

     启动8卡训练。

      ```
      1.安装环境，确认预训练模型放置路径，若该路径路径与model_path默认值相同，可不传参，否则执行训练脚本时必须传入model_path参数；
      2.开始训练
        bash ./test/train_full_8p.sh --data_path=${datasets} --model_path=${pretrain_model}    # 8卡精度
        bash ./test/train_performance_8p.sh --data_path=${datasets} --model_path=${pretrain_model}    # 8卡性能    
      ```
    
   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                          //数据集路径
   --addr                              //主机地址
   --num_workers                       //加载数据进程数      
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小，默认：240
   --lr                                //初始学习率
   --amp                               //是否使用混合精度
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME | Precision |  FPS | Epochs  |  AMP_Type | Torch_Version |
|:----:|:---------:|:----:|:------: | :-------: |:---:|
| 1P-竞品V | -       | - |        1 |       - | 1.5 |
| 8P-竞品V | -       | - |     1200 |       - | 1.5 |
| 1P-NPU-ARM | -         | 20.19 |        1 |       O2 | 1.8 |
| 8P-NPU-ARM | 0.907     |   88.073 |  1200 |       O2 | 1.8 |
| 1P-NPU-非ARM | -         | 20.265 |        1 |       O2 | 1.8 |
| 8P-NPU-非ARM | -    |   113.988 |  1200 |       O2 | 1.8 |


# 版本说明

## 变更

2022.12.23：Readme 整改。

## FAQ

无。

