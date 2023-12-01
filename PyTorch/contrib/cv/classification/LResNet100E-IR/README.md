# LResNet100E-IR for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

LResNet100E-IR是一个目标识别网络模型，该网络提出一种新的损失函数Additive Angular Margin Loss（ArcFace），通过深度卷积神经网络DCNN学习的特征嵌入，可以有效地增强目标识别的判别能力。

- 参考实现：

  ```
  url=https://github.com/TreB1eN/InsightFace_Pytorch
  commit_id=350ff7aa9c9db8d369d1932e14d2a4d11a3e9553
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
  | PyTorch 1.5 | torchvision==0.2.2.post3 |
  | PyTorch 1.8 | torchvision==0.9.1 |
  
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

   用户自行获取 `faces_emore` 原始数据集，将数据集上传到服务器模型源码包根目录的 `data` 目录下并解压。可参考[源码仓](https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/README.md)的方式获取数据集。

   数据集目录结构参考如下所示。

   ```
   data
    |-- data_pipe.py
	|-- faces_emore
            |-- agedb_30
            |-- calfw
            |-- cfp_ff
            |-- cfp_fp
            |-- cfp_fp
            |-- cplfw
            |-- imgs
            |-- lfw
            |-- vgg2_fp             
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   # 创建日志存储，模型存储目录
   rm -rf ./work_space/* 
   mkdir ./work_space/history && mkdir ./work_space/log && mkdir ./work_space/models && mkdir ./work_space/save
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=./data/faces_emore/  # 单卡精度，大约花费40h
     
     bash ./test/train_performance_1p.sh --data_path=./data/faces_emore/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=./data/faces_emore/  # 8卡精度，大约花费7h
     
     bash ./test/train_performance_8p.sh --data_path=./data/faces_emore/  # 8卡性能
     ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh --data_path=./data/faces_emore/lfw.bin --pth_path=real_pre_train_model_path	# 8卡评测
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。
   
   --pth_path参数填写训练权重生成路径，需写到权重文件的一级目录。
   
   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --net_mode                          //网络模型
   --net_depth                         //网络模型深度
   --data_mode                         //数据集模式
   --data_path                         //数据集路径
   --max_iter						   //最大迭代次数
   --start_epoch					   //开始的epoch数      
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小
   --lr                                //初始学习率，默认：0.01
   --num_workers					   //数据集加载进程数
   --device_type					   //训练设备类型
   --device_id						   //训练设备号
   --use_amp                           //是否使用混合精度
   --loss-scale                        //混合精度lossscale大小
   --opt-level                         //混合精度类型
   多卡训练参数：
   --distributed    				   //是否使用多卡训练
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Accuracy | FPS  | Epochs | AMP_Type | Torch_Version | lr | Batch_Size |
| :------: | :---: | :--: | :----: | :------: | :-----------: | :--: | :--: |
| 1p-竞品V | - | - |  2  | - | 1.5 | - | - |
| 8p-竞品V | - | - |  20 | - | 1.5 | - | - |
|  1p-NPU  |    -   | 505.43  | 2  |  O2  |  1.8  | 0.001 | 256   |
|  8p-NPU  | 0.9967 |   4696  | 20 |  O2  |  1.8  | 0.01  | 320*8 |

# 版本说明

## 变更

2022.03.18：首次发布。

## FAQ

无。


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md