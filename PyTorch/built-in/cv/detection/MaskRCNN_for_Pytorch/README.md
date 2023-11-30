# MaskRCNN for Pytorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

MaskRCNN是一个实例分割（Instance segmentation）框架，通过增加不同的分支可以完成目标分类，目标检测，语义分割，实例分割，人体姿态估计等多种任务。

- 参考实现：

  ```
  url=https://github.com/mlcommons/training
  commit_id=2f4a93fb4888180755a8ef55f4b977ef8f60a89e
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/detection/
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.11 | cocoapi torchvision==0.12.0 torchvision_npu==0.12.0|
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》搭建torch环境。
  
- 安装依赖。

  在模型根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```shell
  pip install -r requirements.txt  
  ```
- 编译安装 `cocoapi`。
  ```shell
  git clone https://github.com/cocodataset/cocoapi.git
  cd cocoapi/PythonAPI
  python setup.py build_ext install
  ```
- 安装`torchvision`及`torchvision_npu`。

  请参考《[Torchvision Adapter](https://gitee.com/ascend/vision/tree/v0.12.0-dev/)》编译安装Torchvision及Torchvision Adapter插件（即torchvision_npu）。

- 编译安装`maskrcnn`。
  ```shell
  cd MaskRCNN_for_Pytorch
  python setup.py build develop
  ```
  

## 准备数据集

1. 获取数据集。

   用户自行获取原始 `COCO` 数据集，将数据集上传到服务器任意路径下并解压。

   数据集目录结构参考如下所示。

   ```
    ├── coco2017
    │   ├── annotations
    │          ├── instances_train2017.json
    │          ├── instances_val2017.json
    │          ├── ......   
    │   ├── train2017
    │          ├── 000000000009.jpg
    │          ├── 000000000025.jpg
    │          ├── ......
    │   ├── val2017
    │          ├── 000000000139.jpg
    │          ├── 000000000285.jpg
    │          ├── ......             
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。


## 获取预训练模型

模型脚本会自动下载预训练权重文件。若下载失败，请自行准备 `R-50.pkl` 权重文件，并修改`configs/e2e_mask_rcnn_R_50_FPN_1x.yaml`中的`MODEL.WEIGHT`的值为权重路径。

# 开始训练

## 训练模型


1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机8卡训练。

   - 单机8卡训练
   
     ```shell
     bash test/train_full_8p.sh  --data_path=/data/xxx/ # 8卡精度
     ```
   --data_path参数填写数据集路径，需写到数据集的一级目录。

   训练完成后，权重文件保存在`test/output`路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME        | FPS    | Box mAP | Segm mAP |
| ----------- | ------ | ---------------------- | -------------- |
 8p-竞品V | 56 | 0.378                | 0.343        |
 8p-NPU-910 | 43 | 0.377                | 0.342        |


# 公网地址说明
无。

# 版本说明

## 变更

2023.09.02：首次发布。

## FAQ

无。
   
# 公网地址说明

代码涉及公网地址参考 public_address_statement.md