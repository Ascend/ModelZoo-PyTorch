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
  | 硬件 | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | NPU固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，下载 ILSVRC2015-VID 数据集和 OTB2015 数据集。
   
   数据集目录结构参考：
   ```
   ├── data  
   │    ├── ILSVRC_VID_CURATION  
   │    ├── ILSVRC_VID_CURATION.lmdb  
   ├── dataset 
   │    ├── OTB 
   ```


2. 数据预处理。
	- 首先，运行 bin/create_dataset.py 和 bin/create_lmdb.py 对 ILSVRC2015-VID 数据集进行预处理。
	- real_data_path 是 ILSVRC2015-VID 数据集所在的位置。
	- out_data_path 是预处理图像的位置。
	- lmdb_data_path 必须是 out_data_path+".lmdb"，例如如果 out_data_path 是 "./data/ILSVRC_VID_CURATION"，那么 lmdb_data_path 必须是 "./data/ILSVRC_VID_CURATION.lmdb"。
	
	```
	python3.7 bin/create_dataset.py --d real_data_path --o out_data_path 
	python3.7 bin/create_lmdb.py --d out_data_path --o lmdb_data_path
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
     # 训练 1p 性能
     bash ./test/train_performance_1p.sh --data_path = out_data_path    
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     # 训练 8p 精度
     bash ./test/train_full_8p.sh --data_path = out_data_path
     
     # 训练 8p 性能  
     bash ./test/train_performance_8p.sh --data_path = out_data_path
     
     # 测试 8p 精度
     bash ./test/train_eval_8p.sh --data_path = real_data_path
     ```

   --data\_path参数填写数据集路径。

模型训练脚本参数说明如下。

```
公共参数：
--data                              //数据集路径
--workers                           //加载数据进程数      
--epoch                             //重复训练次数
   ```
   
日志路径：

test/output/device_id/train_${device_id}.log # 训练详细日志

test/output/device_id/siamfc_bs32_8p_fps_loss.log # 8p 训练损失结果日志

test/output/device_id/siamfc_bs32_8p_acc.log # 8p 训练准确度结果日志

test/output/1p_perf/device_id/siamfc_bs32_1p_fps_loss.log  # 1p训练性能结果日志

test/output/8p_perf/device_id/siamfc_bs32_8p_fps_loss.log # 8p训练性能结果日志

test/output/output.prof  # 1p 训练 prof 文件

# 训练结果展示

**表 2**  训练结果展示表

| NAME | Precision |  FPS  | Npu_nums | Epochs | AMP_Type |
|:-------:| :-------: | :--: | :------: | :----: | :------: |
| 1p_1.5 |     -     |950 |    1     |   1    |    O2    |
| 8p_1.5 |  0.756  |  3660 |    8     |   50   |    O2    |
| 1p_1.8 |     -     |  1097.249 |    1     |   1    |    O2    |
| 8p_1.8 |  0.751  |  5941.962 |    8     |   50   |    O2    |


# 版本说明

## 变更

2022.12.05：更新torch1.8版本，重新发布。

2020.07.08：首次发布。

## 已知问题

无。











