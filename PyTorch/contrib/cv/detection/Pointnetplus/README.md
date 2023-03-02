# PointNetPlus for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

PointNet++使用多层次特征提取结构，先在输入点集中选择一些点作为中心点，然后围绕每个中心点选择周围的点组成一个区域，之后每个区域作为PointNet的一个输入样本，得到一组特征，这个特征就是这个区域的特征。之后中心点不变，扩大区域，把上一步得到的那些特征作为输入送入PointNet，以此类推，这个过程就是不断的提取局部特征，然后扩大局部范围，最后得到一组全局的特征，然后进行分类。

- 参考实现：

  ```
  url=https://github.com/yanx27/Pointnet_Pointnet2_pytorch
  commit_id=768cd018b73c5e358f7783fec32140f6a687b133
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
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括modelnet40等，将数据集上传到服务器任意路径下并解压。

   以modelnet40数据集为例，数据集目录结构参考如下所示。

   ```
   ├── modelnet40
         ├──类别1
            │──txt1
            │──txt2
            │   ...
         ├──类别2
            │──txt1
            │──txt2
            │   ...
         ├──...

         ├──train.txt

         ├──test.txt
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
   --data                              //数据集路径
   --use_cpu                           //是否使用cpu训练
   --npu							    //使用的npu设备id
   --epoch                             //重复训练次数
   --batch_size                        //训练批次大小
   --learning_rate                     //初始学习率，默认：0.001
   --workers                           //加载的线程数
   --dist_backend                      //分布式后端
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|  NAME  | Acc@1  |  FPS   | Epochs | AMP_Type | Torch_version |
| :----: | :----: | :----: | :----: | :------: | :-----------: |
| 1p-竞品V |   -    | 27.3 |   1    |    -    |      1.5      |
| 8p-竞品V | 92.1 | 147.2 |  200   |    -   |      1.5      |
| 1p-NPU |   -    | 12.295 |   1    |    O2    |      1.8      |
| 8p-NPU | 91.670 | 60.527 |  200   |    O2    |      1.8      |

# 版本说明

## 变更

2023.03.02：更新readme，重新发布。

2022.06.08：首次发布。

## FAQ

无。