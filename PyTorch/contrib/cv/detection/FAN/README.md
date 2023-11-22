# FAN for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

FAN是一个目标对齐检测网络，通过对目标标志的检测，既能够检测2D也能够检测3D坐标中的点。

- 参考实现：

  ```
  url=https://github.com/1adrianb/face-alignment
  commit_id=c49ca6fef8ffa95a0ac7ce698e0b752ac91f6d42
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/detection
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch1.5 | scipy>=0.17.0 |
  | PyTorch1.8 | scipy>=0.17.0 |
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r requirements.txt
  ```

## 准备数据集

   用户自行获取原始数据集，可选用的开源数据集包括300-W等，进入到项目根目录并创建`dataset`目录，将数据集放在`dataset`下（数据集包含4个子数据集共7674张图片）：

   ```
   # $FAN_ROOT 为项目根目录
   $FAN_ROOT/dataset/
   ```

   以300-w数据集为例，数据集目录结构参考如下所示。

   ```
    dataset
       └── ibug_300W_large_face_landmark_dataset
           ├── aww
           ├── bug
           ├── Helen
           │   ├── trainset
           │   └── testset
           └── lfpw
               ├── trainset
               └── testset
   ```

   > **说明：**
   > 该数据集的训练过程脚本只作为一种参考示例。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=/data/xxx/ --landmarks_type=关键点类型"2D"或"3D"  # 单卡精度

     bash ./test/train_performance_1p.sh --data_path=/data/xxx/ --landmarks_type=关键点类型"2D"或"3D"  # 单卡性能
     ```


   --data_path参数填写数据集路径，需写到数据集的一级目录。

   --landmarks_type参数填写关键点类型"2D"或"3D"。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                              //数据集路径
   --landmarks_type                         //关键点类型
   --steps                                  //重复训练次数
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

   关键点坐标，图像数据保存路径如下：
   ```
    #2D关键点坐标保存路径如下
    $FAN_ROOT/result/points/2D_npu.npy
    #3D关键点坐标保存路径如下
    $FAN_ROOT/result/points/3D_npu.npy
    #2D关键点图像保存路径如下
    $FAN_ROOT/result/images/2D/
    #3D关键点图像保存路径如下
    $FAN_ROOT/result/images/3D/
   ```

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | FPS  | Epochs | Landmarks_Type | Torch_Version |
|  :---:   | :--: | :----: | :------: | :-----------: |
| 1p-竞品V |  0.62 | 50  |  2D   |  1.5  |
| 1p-竞品V |  0.49 | 50  |  3D   |  1.5  |
|  1p-NPU  |  0.59 | 50  |  2D  |   1.5  |
|  1p-NPU |  0.47  | 50  |  3D  |   1.5  |

# 版本说明

## 变更

2020.10.14：更新内容，重新发布。

2020.07.08：首次发布。

## FAQ

无。


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md