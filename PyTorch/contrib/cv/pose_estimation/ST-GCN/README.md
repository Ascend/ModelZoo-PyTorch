# ST-GCN for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述
动态骨架模型ST-GCN，它可以从数据中自动地学习空间和时间的patterns，这使得模型具有很强的表达能力和泛化能力。在Kinetics和NTU-RGBD两个数据集上与主流方法相比，取得了质的提升。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmskeleton/tree/master/deprecated/origin_stgcn_repo
  commit_id=b4c076baa9e02e69b5876c49fa7c509866d902c7
  ```
- 适配昇腾 AI 处理器的实现：
  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/pose_estimation
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

    **表 1** 版本配套表

       | 配套      | 版本                                                                           |
       |------------------------------------------------------------------------------| ------------------------------------------------------------ |
       | 硬件      | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)  |
       | NPU固件与驱动  | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)  |
       | CANN    | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
       | PyTorch | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)                       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户[从作者的GoogleDrive](https://gitee.com/link?target=https%3A%2F%2Fdrive.google.com%2Fopen%3Fid%3D103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb)下载 Kinetics-skeleton 数据集原始数据集，将数据集上传到服务器任意路径下并解压。
   
   数据集目录结构参考如下所示。
   
   ```
   ├── Kinetics
         ├──kinetics-skeleton
              ├──train_data.npy     
              ├──train_label.pkl
              ├──val_data.npy
              ├──val_label.pkl 
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
     bash ./test/train_full_1p.sh --data_path={data/path} # train accuracy
     ```
     测试单卡性能。
     ```
     bash ./test/train_performance_1p.sh --data_path={data/path} # train performance
     
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path={data/path} # train accuracy
     ```
      测试8卡性能。
     ```
     bash ./test/train_performance_8p.sh --data_path={data/path} # train performance
     ```

   - 启动单卡评估

     ```
     bash ./test/train_eval_1p.sh --data_path={data/path}
     ```

   - onnx转换

     ```
     python3.7.5 pthtar2onnx.py
     ```

   --data_path参数填写数据集路径。

3. 模型训练脚本参数说明如下。

   ```
   --device                设备卡号
   --data_path             数据集路径
   --use_gpu_npu           使用的设备，npu或gpu
   --lr                    学习率
   --train_epochs          训练epoch
   --test_path_dir    		模型保存路径
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示



**表 2** 训练结果展示表
    
| NAME      | Acc@1 |     FPS | Epochs | AMP_Type |
| --------- | ----- | ------: | ------ | -------: |
| NPU1.5-1P | 31.62 |      46 | 50     |       O2 |
| NPU1.5-8P | 31.62 |     293 | 50     |       O2 |
| NPU1.8-1P | -     | 373.333 | 2      |       O2 |
| NPU1.8-8P | 32.33 | 978.645 | 50     |       O2 |



# 版本说明

## 变更

2022.11.14：更新内容，重新发布。



## 已知问题



无。
