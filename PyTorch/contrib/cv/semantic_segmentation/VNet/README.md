# VNet for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

V-Net是一个早期的全卷积的三维图像分割网络，基本网络架构与2D图像分割网络U-Net相似，为了处理3D医学图像，采用了3D卷积模块和3D转置卷积模块。

- 参考实现：

  ```
  url=https://github.com/mattmacy/vnet.pytorch
  commit_id=a00c8ea16bcaea2bddf73b2bf506796f70077687
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/semantic_segmentation
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}      # 克隆仓库的代码
  cd {code_path}       # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
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
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

这里提供已预处理数据的下载链接: https://pan.baidu.com/s/1Vg8e6UISiWhpjsabSHCuew?pwd=55mc; 或者可按照如下步骤进行数据获取和处理。

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括[LUNA16](https://luna16.grand-challenge.org/Download/)，将数据集上传到服务器任意路径下并解压。

   以LUNA16数据集为例，数据集目录结构参考如下所示。

   ```
   ├── LUNA16
         ├──lung_ct_image
              ├──1.3.6.1.4.1.14519.5.2.1.6279.6001.997611074084993415992563148335.mhd                    
              ├──...                     
         ├──seg-lungs-LUNA16
              ├──1.3.6.1.4.1.14519.5.2.1.6279.6001.997611074084993415992563148335.mhd
              ├──...    
         ├──normalized_lung_ct
              ├──1.3.6.1.4.1.14519.5.2.1.6279.6001.997611074084993415992563148335.mhd                    
              ├──...                     
         ├──normalized_lung_mask
              ├──1.3.6.1.4.1.14519.5.2.1.6279.6001.997611074084993415992563148335.mhd
              ├──...                                      
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理。
    ```
    cd /${模型文件夹名称}
    python3 normalize_dataset.py root_path vox_spacing Z_MAX Y_MAX X_MAX
    ```
    

<!-- ## 获取预训练模型（可选）

请参考原始仓库上的README.md进行预训练模型获取。将获取的bert\_base\_uncased预训练模型放至在源码包根目录下新建的“temp/“目录下。 -->

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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/    
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/   
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                              //数据集路径
   --seed                              //随机种子
   --workers                           //加载数据进程数      
   --lr                                //初始学习率，默认：0.001
   --lr_decay                          //学习率衰减，默认：0.3
   --world_size                        //节点数目，默认：1
   --weight_decay                      //权重衰减，默认：1e-8
   --device                            //设备，默认：'npu'
   --nEpochs                           //重复训练次数
   --batchSz                           //训练批次大小
   --momentum                          //动量，默认：0.9
   --amp                               //是否使用混合精度
   --loss-scale                        //混合精度lossscale大小
   --opt-level                         //混合精度类型
   --device_id                         //多卡训练指定训练用卡
   --save                              //保存路径，例如 model_8p
   多卡训练参数：
   --device_num                        //卡数，默认：8
   --dist_backend                      //并行后端，默认：'hccl'
   --distributed                       //是否使用多卡训练
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。
    ```
    ./output/devie_id/train_${device_id}.log  # training detail log
    ./output/devie_id/train_VNet_bs4_8p_acc_loss.txt  # 8p training performance result log
    ./output/devie_id/VNet_bs4_8p_acc.log  # 8p training accuracy result log
    ./model_8p/vnet_checkpoint.pth.tar  # last checkpoits
    ./model_8p/vnet_model_best.pth.tar # best checkpoits
    ```

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Error rate |  FPS | Epochs | AMP_Type |
| ------- | ----- | ---: | ------ | -------: |
| 1p-NPU1.5 | 0.414%    |  17.93 | 1      |        O2|
| 1p-NPU1.8  | --   | 38.79 | 1      |       O2 |
| 8p-NPU1.5 | 0.745% | 123.15 | 200    |     O2 |
| 8p-NPU1.8  | 0.428% | 218.50 | 200    |       O2 |


# 版本说明

## 变更

2022.08.25：更新内容，重新发布。

2020.07.08：首次发布。

## 已知问题

无。











