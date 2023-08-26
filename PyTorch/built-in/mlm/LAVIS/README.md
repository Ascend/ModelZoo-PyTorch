# LAVIS

-   [概述]()
-   [准备训练环境]()
-   [开始训练]()
-   [训练结果展示]()
-   [版本说明]()

# 概述

## 简述

LAVIS 是一个多模态模型套件，包含CLIP、ALBEF、BLIP、BLIP2、InstructBLIP等多种多模态模型，以及Image-text Retrieval、Image Captioning等下游任务的训练与推理，可用于图文问答、图文检索、图像分类等任务。

- 参考实现：

  ```
  url=https://github.com/salesforce/LAVIS
  commit_id=f982acc73288408bceda2d35471a8fcf55aa04ca
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/mlm/LAVIS
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version  | Java_Version |
  | :------------: | :----------: |
  | PyTorch 1.11.1 | JDK 1.8以上  |

  评估使用 Stanford CoreNLP工具，该工具为Java开发，因此需要Java相关依赖。

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》搭建torch环境。

- 安装依赖。

  在模型根目录下执行命令，安装模型对应PyTorch版本需要的依赖。

  ```shell
  pip install -e .                    # 安装LAVIS以及所需的依赖
  ```

- 安装Megatron-LM，[参考链接](http://gitee.com/ascend/Megatron-LM)。

## 适配情况

| 任务             | 支持模型 | 支持数据集 |
| ---------------- | -------- | ---------- |
| Image Captioning | BLIP2    | COCO2014   |

## BLIP2

### Image Captioning 微调任务

#### 准备数据集

1. 修改数据集默认路径。

   修改 `lavis/configs/default.yaml` 文件的第10行，将 `cache_root` 更改为期望数据集存放的路径。

2. 获取数据集。

   以coco2017数据集为例，用户可以通过执行如下脚本进行下载：

   ```
   cd lavis/datasets/download_scripts/ && python download_coco.py
   ```

   数据结构如下：

   ```
   $dataset
   ├── coco
   	├── images
   		├── train2014
           └── val2014
       └── annotations
           ├── coco_karpathy_test_gt.json
           ├── coco_karpathy_val_gt.json
           └── coco_karpathy_train_gt.json
   └── coco_gt
       ├── coco_karpathy_test_gt.json
       └── coco_karpathy_val_gt.json
   ```

   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。

#### 获取预训练模型

联网情况下，预训练模型会自动下载。

无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

```
facebook/opt-2.7b
bert-base-uncased
```

当模型下载到本地后，需要做如下操作

1. 对于bert-base-uncased模型，需要修改`lavis/models/blip2_models/blip2.py`文件的32、48和55行，将其改为本地模型权重路径；
2. 对于facebook/opt-2.7b模型，需要修改`lavis/models/blip2_models/blip2_opt.py`文件的85、87行，将其改为本地模型权重路径；

#### 开始训练

##### 训练模型

运行训练脚本。

该模型支持单机8卡训练。

- 单机8卡精度

  ```shell
  bash test/train_full_8p_blip2_caption_coco_opt2.7b_ft.sh  # 8卡精度
  ```

* 单机8卡性能

  ```shell
  bash test/train_performance_8p_blip2_caption_coco_opt2.7b_ft.sh # 8卡性能
  ```

  训练完成后，精度最优的权重文件保存在`lavis/output`路径下，并输出模型训练精度信息。

#### 训练结果展示

**表 2**  训练结果展示表

|    NAME     | FPS  | Epoch | batch_size | Bleu@4 | CIDEr |
| :---------: | :--: | :---: | :--------: | :----: | :---: |
|  8p-竞品A   |   102.28   |   5   |     16     |    0.423    |    1.439   |
| 8p-NPU-910B |   61.95   |   5   |     16     |    0.421    |    1.432   |

#### 常见问题

1. 模型首次评估时候，需要下载Standford CoreNLP工具，所需时间较长。

2. 在模型评估的时候，如果出现 `subprocess.calledprocesserror: command '[subprocess.calledprocesserror'java', '-jar', '-xmx8g', 'spice-1.0.jar', <some more commands> -subset -silent returned non-zero exit status 1.` 报错，需要做如下处理

   * 找到pycocoevalcap的安装路径，然后`cd pycocoevalcap/spice/`

   * 在`spice.py`文件的68-73行，做如下更改

     ```python
     spice_cmd = ['java', '-jar', '-Xmx8g', SPICE_JAR, in_file.name, 
     # '-cache', cache_dir, 将该行注释
     '-out', out_file.name,
     '-subset',
      '-silent'
     ]
     ```

# 版本说明

## 变更

2023.08.18：首次发布。
