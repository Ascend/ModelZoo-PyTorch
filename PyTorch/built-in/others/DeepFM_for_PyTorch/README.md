# DeepFM for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

DeepFM是一个最大点击率推荐网络。DeepFM结合了基于推荐系统的因子分解机，以及深度神经网络，用于提取低阶和高阶特征交互。相比于最新的Wide&Deep模型，DeepFM的Wide及Deep部分共享输入，从而无需除了原始特征之外的特征工程。实验证明，在基于benchmark数据以及商业数据的CTR预估任务中，DeepFM展示了良好的有效性以及效率。

- 参考实现：

  ```
  url=https://github.com/shenweichen/DeepCTR-Torch
  commit_id=b4d8181e86c2165722fa9331bc16185832596232
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/others
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

   用户自行下载 `criteo display advertising challenge` 数据集，将数据集上传到服务器任意路径并解压。

   数据集的目录结构参考如下。

   ```
   ├── criteo
   │    ├──criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz # 原始数据集文件       
   │    ├──train.txt                                                           # 解压后生成
   │    ├──test.txt                                                            # 解压后生成
   │    ├──readme.txt                                                          # 解压后生成   
   ```

   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。


2. 数据预处理。
   - 将源码包根目录下的 `criteo_preprocess.py` 拷贝至数据集路径
    
   - 进入数据集目录并执行如下命令：

     ```
     python3 criteo_preprocess.py train.txt
     ```

   - 处理完成之后会在同级目录生成“deepfm_trainval.txt”，“deepfm_test.txt”以及"data.ini"三个数据文件，此时预处理后的数据集目录结构如下所示。

   ```
   ├── criteo
   │    ├──criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz # 原始数据集文件       
   │    ├──train.txt                                                           # 解压后生成
   │    ├──test.txt                                                            # 解压后生成
   │    ├──readme.txt                                                          # 解压后生成
   │    ├──deepfm_trainval.txt                                                 # 预处理后生成
   │    ├──deepfm_test.txt                                                     # 预处理后生成
   │    ├──data.ini                                                            # 预处理后生成
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
   --seed                                  //初始化训练的种子，默认：1234
   --use_npu                               //使用npu，默认：False
   --use_cuda                              //使用cuda，默认：False
   --device_id                             //设备id      
   --dist                                  //分布式训练
   --init_checkpoint                       //初始化checkpoint
   --batch-size                            //训练批次大小
   --lr                                    //初始学习率，默认：0.001
   --amp                                   //是否使用混合精度
   --loss_scale                            //混合精度lossscale大小
   --opt_level                             //混合精度类型
   --data_path                             //数据集路径
   --optim                                 //模型优化器
   --test_size                             //测试时使用的数据大小
   --epoches                               //重复训练次数
   --steps                                 //训练的步数
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息（保存在output目录下）。

# 训练结果展示

**表 3**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V | - | - | 3 | - | 1.5 |
| 8p-竞品V | - | - | 3 | - | 1.5 |
| 1p-NPU | 0.8012 | 10808.5112 | 3 | - | 1.8 |
| 8p-NPU | 0.8018 | 208204.718 | 3 | - | 1.8 |

# 版本说明

## 变更

2022.11.09：更新torch1.8版本，重新发布。

2020.10.14：首次发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 ```./public_address_statement.md```
