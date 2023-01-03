# FairMOT for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

FairMOT是一阶段多目标跟踪器（one-shot MOT），检测模型和Re-ID重识别模型同时进行，提升了运行速率。FairMOT采用 anchor-free 目标检测方法（CenterNet），估计高分辨率特征图上的目标中心和位置；同时添加并行分支 来估计像素级 Re-ID 特征，用于预测目标的 id。

- 参考实现：

  ```
  url=https://github.com/ifzhang/FairMOT
  branch=master 
  commit_id=815d6585344826e0346a01efd57de45498cfe52b
  ```
  
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/detection
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}          # 克隆仓库的代码
  cd {code_path}     	  # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 硬件       | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial   ) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

- 首先创建一个数据集目录dataset，数据集放在这个文件夹下。

-  下载[MOT17数据集](https://motchallenge.net/data/MOT17.zip)
- 下载得到MOT17.zip 解压，然后将数据集处理成如下的文件结构

```
MOT17
|——————images
|        └——————train
|        └——————test
└——————labels_with_ids
         └——————train(empty)
```

- 接下来需要生成标注文件，需要先修改/FairMOT/src/gen_labels_16.py
  将这个文件的seq_root 修改为dataset文件夹的目录+'/MOT17/images/train' 例如：/root/dataset/MOT17/images/train

- 然后将label_root 修改为dataset文件夹的目录+'MOT16/labels_with_ids/train' 例如/root/dataset/MOT17/labels_with_ids/train
  然后执行 

```
python3.7 gen_labels_16.py
```

- 下载 https://github.com/ifzhang/FairMOT 模型，将刚下载下的FairMOT/src下的data文件夹放至本模型的src目录下。

## 准备预训练权重

下载[DLA-34 official]：CenterNet (Objects as Points) 作者在其[github项目](https://github.com/ifzhang/FairMOT)中提供了多种训练后的[模型文件下载（google drive](https://drive.google.com/file/d/1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT/view)），但目前下载受限，[百度网盘下载](https://pan.baidu.com/share/init?surl=yaoRNyyn0EKG48rFJR5cIQ)（提取码：v9h2）

然后放到/FairMOT/models/文件夹下。


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机8卡训练。

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/ 
     ```
     
   - 单机单卡性能
   
     启动单机性能
   
     ```
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/ 
     ```
     
   - 单机8卡性能
   
     启动8卡性能
     
     ```
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/ 
     ```
   
   --data_path参数填写数据集路径。
   
   
   
   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   mot 								//任务名
   --load_model 						//预训练模型如：'../models/ctdet_coco_dla_2x.pth' 
   --data_cfg 							//指定数据配置文件   
   --world_size  						//加载数据进程数
   --batch_size  						//批次大小
   --rank 								//npu设备好
   --lr 30e-4  						//初始学习率，默认：1e-4
   --use_npu True 						//是否启用npu训练
   --use_amp True 						//是否启用amp
   --num_epochs                    	//重复训练次数
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME   | MOTA | FPS     | Torch_version |
| ------ | ---- | ------- | ------------- |
| 1p-NPU | -    | 3.8     | Torch1.5      |
| 1p-NPU | -    | 5.7818  | Torch1.8      |
| 8p-NPU | 84.8 | 28      | Torch1.5      |
| 8p-NPU | 85.2 | 38.2117 | Torch1.8      |



# 版本说明

2022.11.24：更新pytorch1.8版本，重新发布。

2021.10.16：首次发布。

## 已知问题


无。











