# Centroids-Reid for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

图像检索任务包括从一组图库(数据库)图像中查找与查询图像相似的图像。此类系统用于各种应用，例如人员重新识别(ReD)或视觉产品搜索。尽管检索模型得到了积极的发展，但它仍然是一项具有挑战性的任务，这主要是由于视角、光照、背景杂波或遮挡变化引起的类内方差较大，而类间方差可能相对较低。目前，很大一部分研究集中在创建更健壮的特征和修改目标函数上，通常基于三重损失。一些作品尝试使用类的形心/代理表示来缓解计算速度和与三元组损失一起使用的硬样本挖掘的问题。然而，这些方法仅用于训练，在检索阶段被丢弃。在本文中，我们建议在训练和检索过程中使用平均质心表示。这种聚集表示对异常值更为稳健，并确保了更稳定的特征。由于每个类都由一个嵌入表示，即类质心，因此检索时间和存储要求都显著降低。由于降低了候选目标向量的数量，聚合多个嵌入导致了搜索空间的显著减少，这使得该方法特别适合于生产部署。在两个ReID和时尚检索数据集上进行的综合实验证明了该方法的有效性，优于现有技术。

- 参考实现：

  ```
  url=https://github.com/mikwieczorek/centroids-reid 
  commit_id=a1825b7a92b2a8d5e223708c7c43ab58a46efbcf
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

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  |硬件        | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)
  | NPU固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)|
  
- 环境准备指导。

  请参考[《Pytorch框架训练环境准备》](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)。

- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

获取数据集。

   在数据集DukeMTMC-reID实现对Centroids-reid的训练。
   用户自行获取原始数据集，可选用的开源数据集DukeMTMC-reID，将数据集上传到在源码包根目录下新建的“data/”目录下并解压。
   以DukeMTMC-reID数据集为例，数据集目录结构参考如下所示。

   ```
   ├── data
         ├──DukeMTMC-reID
              ├──bounding_box_test/    
              ├──bounding_box_train/
              ├──...                                 
   ```
   
## 获取预训练模型
   下载[resnet50-19c8e357.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth)权重文件，并放到在源码包根目录下新建的“models/”文件夹下，“models/”文件夹遵循以下的目录结构。

   ```
   |-- models
   |   |-- resnet50-19c8e357.pth
   ```

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡性能和单机8卡训练。

   - 单机单卡性能

     启动单卡性能。

     ```
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/    
     ```
     
    - 单机8卡训练
 
      启动8卡训练。
 
      ```
      bash ./test/train_full_8p.sh --data_path=/data/xxx/  
      ```
 
    - 单机8卡性能
 
      启动8卡性能。
 
      ```
      bash ./test/train_performance_8p.sh --data_path=/data/xxx/  
      ```
 
   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下：

   ```
   --GPU_IDS                              //指定训练用卡
   --DATASETS.NAMES                       //数据集名称
   --DATASETS.ROOT_DIR                    //数据集根目录
   --SOLVER.IMS_PER_BATCH                 //训练批次大小
   --SOLVER.MAX_EPOCHS                    //训练最大的epoch数
   --TEST.IMS_PER_BATCH                   //测试批次大小
   --SOLVER.BASE_LR                       //初始学习率
   --OUTPUT_DIR                           //输出目录
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| pytorch版本 | NAME    | 性能 |  是否收敛 | MAP |
| ------- | ----- | --- | ------ |-----|
| pytorch1.5 | 1p-NPU  | 47it/s |  是 | 0.96056 |
| pytorch1.5 | 8p-NPU  | 287it/s |  是 | 0.95472 |
| pytorch1.8 | 1p-NPU  | 87it/s |  是 | -       |
| pytorch1.8 | 8p-NPU  | 879it/s |  是 | 0.96407 |

# 版本说明

## 变更
2022.11.08：更新torch1.8版本，重新发布。

2022.08..24：首次发布。

## 已知问题

无。











