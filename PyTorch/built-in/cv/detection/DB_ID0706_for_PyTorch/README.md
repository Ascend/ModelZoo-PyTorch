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
  | 硬件    | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)  |
  | 固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)  |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)                       |

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

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 请用户自行下载icdar2015数据集，解压放在任意文件夹datasets下;

    ```
    /datasets
       |--icdar2015
    ```
2. 下载预训练模型[MLT-Pretrain-Resnet50]( https://drive.google.com/open?id=1T9n0HTP3X3Y_nJ0D1ekMhCQRHntORLJG )，放置到path-to-model-directory文件夹中;

    ```
    /path-to-model-directory
       |-- MLT-Pretrain-ResNet50
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

   单卡训练流程：

    ```
    1.安装环境，确认预训练模型放置路径，若该路径路径与model_path默认值相同，可不传参，否则执行训练脚本时必须传入model_path参数；
    2.开始训练
        bash ./test/train_full_1p.sh --data_path=${datasets} --model_path=预训练模型路径    # 精度测试
        bash ./test/train_performance_1p.sh --data_path=${datasets} --model_path=预训练模型路径    # 性能测试
        [ data_path为数据集路径，写到datasets，即data_path路径不包含icdar2015 ]   
    ```
    **注意**：如果发现打屏日志有报checkpoint not found的warning，请再次检查预训练模型MLT-Pretrain-Resnet50的配置，以免影响精度。

    多卡训练流程：

    ```
    1.安装环境，确认预训练模型放置路径，若该路径路径与model_path默认值相同，可不传参，否则执行训练脚本时必须传入model_path参数；
    2.开始训练
        bash ./test/train_full_8p.sh --data_path=${datasets} --model_path=预训练模型路径    # 精度测试
        bash ./test/train_performance_8p.sh --data_path=${datasets} --model_path=预训练模型路径    # 性能测试
        [ data_path为数据集路径，写到datasets，即data_path路径不包含icdar2015 ]    
    ```
    
    模型评估：
    
    ```
    执行脚本 bash eval_precision.sh
    ```
   
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

# 训练结果展示

**表 2**  训练结果展示表

| NAME | Precision |  FPS | Npu nums |  AMP_Type |
|-----|-----------|-----:| ------   | -------: |
| NPU | -         | 10.5 |        1 |       O2 |
| NPU | 0.907     |   61 |        8 |       O2 |


# 版本说明

## 变更

2022.12.23：Readme 整改。

## 已知问题

无。

