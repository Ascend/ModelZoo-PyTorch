# DeepMar for PyTorch\_Owner

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

DeepMar是一个深度多属性联合学习模型，采用ResNet50做主干网络，把行人属性识别当做了多标签分类问题，并且能够很好地利用属性之间的关联关系。

- 参考实现：

  ```
  url=https://github.com/dangweili/pedestrian-attribute-recognition-pytorch.git
  commit_id=468ae58cf49d09931788f378e4b3d4cc2f171c22
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
  ```

- 通过Git获取代码方法如下：

  ```
  git clone {url}        # 克隆仓库的代码
  cd {code_path}         # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 所需依赖。

  **表 2**  依赖列表

  | 依赖名称    | 版本             |
  | ----------- | ---------------- |
  | torch       | 1.8.1+ascend.rc2 |
  | torch-npu   | 1.8.1rc2         |
  | Pillow      | 9.1.0            |
  | apex        | -                |
  | torchvision | 0.9.1            |


- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖（根据模型需求，按需添加所需依赖）。

  ```
  pip install -r requirements.txt
  ```

## 准备数据集

1. 获取数据集。

   模型所需数据集为[peta](https://pan.baidu.com/s/1q8nsydT7xkDjZJOxvPcoEw)，密码: 5vep。数据集目录结构参考如下所示。

   ```
   ├── peta
         ├──PETA.mat                 
         ├──images  
              ├── 图片1
              ├── 图片2
              ├── ...
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
     bash ./test/train_full_1p.sh --data_path=./dataset/peta/    
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=./dataset/peta/  
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --save_dir                          // 数据集路径
   --addr                              // 主机地址
   --npu                               // 单卡训练指定训练用卡号
   --workers                           // 加载数据进程数，默认：2 
   --total_epochs                      // 重复训练次数，默认150
   --batch_size                        // 训练批次大小
   --new_params_lr                     // 学习率，默认：0.001
   --finetuned_params_lr               // 最终学习率，默认：0.001
   --steps_per_log                     // 打印间隔， 默认是20
   --multiprocessing-distributed       // 是否使用多卡训练
   --device-list                       // 多卡训练指定训练用卡号，默认值：'0,1,2,3,4,5,6,7'
   --amp                               // 是否使用混合精度
   --loss_scale                        // 混合精度lossscale大小
   --opt_level                         // 混合精度类型
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 3**  训练结果展示表

| NAME    | Acc@1 |     FPS | Epochs | AMP_Type |
| ------- | ----- | ------: | ------ | -------: |
| 1p-竞品 | -     |       - | -      |        - |
| 1p-NPU  | -     | 656.556 | 150    |       O2 |
| 8p-竞品 |       |       - | -      |        - |
| 8p-NPU  | 76.52 | 4806.53 | 150    |       O2 |


# 版本说明

## 变更

2022.08.17：更新内容，重新发布。

2022.03.08：首次发布。

## 已知问题


无。