# Lifespan_ID2972_for_pytorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)




## 简介
Lifespan Age Transformation Synthesis 是一种基于GAN的方法，用于从单个输入人脸图像模拟连续老化过程。
- 参考论文：https://arxiv.org/pdf/2003.09764.pdf
  

- 参考实现：
  ```
  url=https://github.com/royorel/Lifespan_Age_Transformation_Synthesis
  ```

- 适配昇腾 AI 处理器的实现：
  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/others
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/master/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```
## 准备数据集

1. 获取FFHQ-Aging数据集. 
    ```
    url=https://github.com/royorel/FFHQ-Aging-Dataset
    ```
2. 数据预处理
    ```
    cd datasets
    python create_dataset.py --folder <path to raw FFHQ-Aging directory> --labels_file <path to raw FFHQ-Aging labels csv file> [--train_split] [num of training images (default=69000)]
    ```
    
3. 如果你希望在你自己的数据集上进行训练&测试，结构应当如下
    ```
    ├── dataset_name                                                                                                                                                                                                       
    │   ├── train<class1> 
    |   |   └── image1.png
    |   |   └── image2.png
    |   |   └── ...                                                                                                
    │   │   ├── parsings
    │   │   │   └── image1.png
    │   │   │   └── image2.png
    │   │   │   └── ...                                                                                                                             
    ...
    │   ├── train<classN> 
    |   |   └── image1.png
    |   |   └── image2.png
    |   |   └── ...                                                                                                
    │   │   ├── parsings
    │   │   │   └── image1.png
    │   │   │   └── image2.png
    │   │   │   └── ... 
    │   ├── test<class1> 
    |   |   └── image1.png
    |   |   └── image2.png
    |   |   └── ...                                                                                                
    │   │   ├── parsings
    │   │   │   └── image1.png
    │   │   │   └── image2.png
    │   │   │   └── ...                                                                                                                             
    ...
    │   ├── test<classN> 
    |   |   └── image1.png
    |   |   └── image2.png
    |   |   └── ...                                                                                                
    │   │   ├── parsings
    │   │   │   └── image1.png
    │   │   │   └── image2.png
    │   │   │   └── ... 
    ```
    
### 获取预训练模型
    python download_models.py

# 开始训练

### Training on NPU
- 训练超参

  - Batch size: 3
  - amp: True(02)
  - lr: 0.001(可修改)
  - Train epoch: 400

```python run.py (modelarts)```

```./run_scripts/train.sh (ascend-torch1.5)```


模型训练脚本参数说明如下。

   ```
   公共参数：
   --name                              //选择训练（male|female）model
   --checkpoints_dir                   //结果路径（./checkpoints）
   --dataroot                          //数据集路径（datasets/males）     
   --epoch                             //重复训练次数（400）
   --batchSize                         //训练批次大小(3)
   --lr                                //adam学习率（0.001）
   --amp                               //是否使用混合精度（True）
   --loss_scale_value                  //混合精度lossscale大小（默认128.0）
   --opt_level                         //混合精度类型（默认O2）
   --npu_ids                           //可用npu设备
   ```
如果需要修改参数可以在如下目录修改
```run_scripts/train/train.sh```

```options/train_options.py```


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已开启混合精度（O2）

优化器使用NpuFusedAdam、combine_grad=True性能提升约20%

##训练结果展示

| NAME    |          -            |  FPS      | Epochs | sec/epoch |   acc    |
| ------  | --------------------- | --------- | ------ | --------  |   ----   |
| NPU_1p  | torch1.5+Ascend910    | 0.001385  | 15     | 2943.1    |   None   |
| GPU_1p  | torch1.5+V100         | 0.001019  | 15     | 2166.1    |   None   |

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md