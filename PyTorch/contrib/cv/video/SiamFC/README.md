# SiamFC for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

SiamFC通过全卷积的孪生网络结构实现了对视频或图像序列中某一特定目标的动向过程。该模型将经典的特征提取网络AlexNet与孪生网络相结合，网络采用全卷积的方式对模板图片与搜索图片进行卷积计算，以在搜索图片上找出最符合模板图片的位置。

- 参考实现：

  ```
  url=https://github.com/HonglinChu/SiamTrackers/tree/master/SiamFC/SiamFC
  commit_id=2dd15d2591d8f34074b3074c0680fbc962c40cc6
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/video
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

   用户自行获取原始数据集，下载 ILSVRC2015-VID 数据集和 OTB2015 数据集。
   
   数据集目录结构参考如下所示。
   ```
   ├── data  
   │    ├── ILSVRC_VID_CURATION  
   │    ├── ILSVRC_VID_CURATION.lmdb  
   ├── dataset 
   │    ├── OTB
   ```
   
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理。
	- 首先，在源码包根目录下运行 bin/create_dataset.py 和 bin/create_lmdb.py 对 ILSVRC2015-VID 数据集进行预处理。
	- real_data_path 是 ILSVRC2015-VID 数据集所在的位置。
	- out_data_path 是预处理图像的位置。
	- lmdb_data_path 必须是 out_data_path+".lmdb"，例如 out_data_path 是 "./data/ILSVRC_VID_CURATION"，那么 lmdb_data_path 则为 "./data/ILSVRC_VID_CURATION.lmdb"。
	
	```
	python3 bin/create_dataset.py --d real_data_path --o out_data_path
	python3 bin/create_lmdb.py --d out_data_path --o lmdb_data_path
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
     bash ./test/train_performance_1p.sh --data_path=out_data_path  # 单卡性能
     ```
     
   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=out_data_path  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=out_data_path  # 8卡性能
     ```
   
   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh
     ```
   
   --data_path参数填写数据集路径，需写到数据集的一级目录（填写数据预处理中 out_data_path 路径信息）。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                              //数据集路径
   --workers                           //加载数据进程数      
   --epoch                             //重复训练次数
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

| NAME | Precision |  FPS  | Epochs | AMP_Type | Torch_Version |
|:-------:| :-------: | :--: | :----: | :------: | :------: |
| 1p-NPU |     -     |  1097.249 |   1    |    O2    |    1.8    |
| 8p-NPU |  0.751  |  5941.962 |   50   |    O2    |    1.8    |


# 版本说明

## 变更

2023.03.13：更新readme，重新发布。

2020.07.08：首次发布。

## FAQ

无。


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md