# GLIP for PyTorch

-   [概述](#1)
-   [准备训练环境](#2)
-   [开始训练](#3)
-   [训练结果展示](#4)
-   [版本说明](#5)

# 概述

## 简述

GLIP是一种用于视觉定位的语言-图像预训练模型，可以学习对象级、语言感知和语义丰富的视觉表示。GLIP统一了预训练的对象检测和短语定位。统一框架带来了两个优点，1）允许GLIP从检测和定位数据中学习，以提高这两项任务的精度并得到一个优秀的定位模型，2）GLIP可以通过自训练的范式利用大量的图文对生成定位框，得到语义丰富的特征。实验证明，GLIP具有强大的零样本、少样本迁移能力。
- 参考实现：
  
  ```bash
    url=https://github.com/microsoft/GLIP/
    commit_id=a5f302bfd4c5c67010e29f779e3b0bde94e89985
  ```

- 适配昇腾 AI 处理器的实现：

  ```bash
    url=https://gitee.com/ascend/ModelZoo-PyTorch.git
    code_path=PyTorch/built-in/others
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  ****表 1**** 版本支持表

  | Torch_Version |                三方库依赖版本                |
  |---------------|:-------------------------------------:|
  | PyTorch 1.11  | mmcv-full==1.7.1; torchvision==0.12.0 |


- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。

  ```bash
  pip install -r requirements.txt
  python setup.py build develop
  ```
- 注意 mmcv-full 需要拉取1.x分支的最新代码源码编译，不要使用pip安装方式。(如果环境中有mmcv，请先卸载再编译安装)

  ```bash
  git clone -b 1.x https://github.com/open-mmlab/mmcv.git
  cd mmcv
  MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
  pip install -r requirements/runtime.txt
  MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
  ```
- 如果环境中自动安装了torch和torchvision，请先卸载再安装正确依赖版本

  
## 准备训练数据集

   使用coco2017数据集。
   准备好数据集后放到 /${模型文件夹名称} 目录下，并重命名为coco

   ```
   ├── coco
         ├── annotations               
         	├── instances_train2017.json
         	├── instances_val2017.json ...
         ├── train2017
         	├── 000000******.jpg ...
         ├── val2017
         	├── 000000******.jpg ...
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。
   

## 准备预训练模型
- 下载预训练模型glip_tiny_model_o365_goldg_cc_sbu.pth，路径为/${模型文件夹名称}/pretrain/glip_tiny_model_o365_goldg_cc_sbu.pth。
- 下载预训练语言模型文件夹bert-base-uncased，路径为/${模型文件夹名称}/bert-base-uncased。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```bash
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练、单机8卡训练。

   + 单机单卡训练

     启动单卡训练：

     ```bash
     bash test/train_full_1p.sh         #单卡训练
     
     bash test/train_performance_1p.sh  #单卡性能测试
     ```
   
   + 单机8卡训练
   
     启动8卡训练：
   
     ```bash
     bash test/train_full_8p.sh          #多卡训练
     
     bash test/train_performance_8p.sh   #多卡性能测试
     ```

   --batch_size                      //训练批次大小

   --load_from                       //加载的预训练参数路径

   --fp32                            //开启fp32模式

   --fp16                            //开启fp16模式


   模型训练脚本参数说明如下。

   ```bash
      --config-file                  //配置文件
      --override_output_dir          //结果保存路径
      --early_stop_iteration         //早停训练迭代数
      MODEL.WEIGHT                   //预训练权重路径
      MODEL.LANGUAGE_BACKBONE.TOKENIZER_PATH   //分词模型路径
      MODEL.LANGUAGE_BACKBONE.MODEL_PATH       //预训练bert权重路径
      SOLVER.IMS_PER_BATCH           //训练批次大小
      SOLVER.USE_AMP                 //使能混精训练
      SOLVER.MAX_EPOCH               //训练epoch数
   ```  
    
   训练完成后，权重文件保存在/${模型文件夹名称}/test/output路径下，并输出模型训练精度和性能信息。
     

# 训练结果展示

**表 2**  训练结果展示表

| NAME   | mAP  |  FPS  | USE AMP | Iterations | Batch Size |Torch_Version |
|--------|:----:|:-----:|---------|------------|------------| ---------------|
| 8p-NPU | 54.5 | 6.941 | False   | 35000      | 8          |1.11          |
| 8p-竞品A | 54.5 | 8.023 | False   | 35000      | 8          |1.9           |
| 8p-NPU | 54.7 | 6.737 | True   | 35000      | 8          |1.11          |
| 8p-竞品A | 54.4 | 7.371 | True   | 35000      | 8          |1.9           |



# 版本说明

## 变更

2023.09.08：首次发布。
## FAQ
无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md 、DATA.md 及 README_RAW.md
