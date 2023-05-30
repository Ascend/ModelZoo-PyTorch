# OpenCLIP for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

open_clip是由OpenAI开源的文图匹配预训练模型，通过在LAION-400M、LAION-2B等大规模文图数据上进行自监督预训练，使模型具备强大的跨模态能力，凭借此能力，open_clip在文图检索、问答等多种下游任务中取得惊人效果。

- 参考实现：

  ```
  url=https://github.com/mlfoundations/open_clip
  commit_id=3b081484c360569179e270016b5549b7686d42ab
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/others
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.11 | - |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  ```
  pip install -r requirements.txt
  python3 -m pip install -e .
  ```

## 准备预训练模型与词表

- 在官网下载CLIP-ViT-B-32-laion2B-s34B-b79K预训练模型
- 下载bpe_simple_vocab_16e6.txt.gz词表文件，并放在src/open_clip路径下

## 准备数据集

1. 获取数据集flickr30k。

1. 使用tools/flickr30k_handle.py切分出训练集与测试集。将flickr30k_handle.py拷贝至数据集目录下并执行，生成flickr30k_test.csv与flickr30k_train.csv

   数据集目录结构参考如下所示。
   
   ```
   ├── flickr30k
      ├──flickr30k-images   
      ├──flickr30k_test.csv                  
      ├──flickr30k_train.csv     
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
     bash test/train_full_1p.sh --data_path=flickr30k数据路径 --pretrain_model=预训练模型路径       # 单卡训练
     
     bash test/train_performance_1p.sh --data_path=flickr30k数据路径 --pretrain_model=预训练模型路径       # 单卡性能
     ```
     
   - 单机8卡训练

     启动8卡训练。
     ```
     bash test/train_full_8p.sh --data_path=flickr30k数据路径 --pretrain_model=预训练模型路径       # 8卡训练
     
     bash test/train_performance_8p.sh --data_path=flickr30k数据路径 --pretrain_model=预训练模型路径       # 8卡性能
     ```
     
     
   
   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --device_id                             //训练卡id指定
   --data_path                             //数据路径
   --pretrain_model                        //预训练模型路径
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | image_to_text_R@5 | FPS  | Epochs | AMP_Type | batch_size |
| :------: | :---------------: | :--: | :----: | :------: | :--------: |
|  1p-NPU  |         -         | 563  |   2    |    O2    |    128     |
| 1p-竞品V |         -         | 544  |   2    |    O2    |    128     |
|  8p-NPU  |       0.777       | 3342 |   20   |    O2    |    1024    |
| 8p-竞品V |       0.779       | 2883 |   20   |    O2    |    1024    |


# 版本说明

## 变更

2023.05.16：首次发布。

## FAQ

无。