# XLM for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

XLM模型是transformer的改进模型，其克服了信息不互通的难题，将不同语言放在一起采用新的训练目标进行训练，从而让模型能够掌握更多的跨语言信息。这种跨语言模型的一个显著优点是，对于预训练后的后续任务（比如文本分类或者翻译等任务），训练语料较为稀少的语言可以利用在其他语料上学习到的信息。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/XLM.git
  commit_id=cd281d32612d145c6742b4d3f048f80df8669c30
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/nlp
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |
  | PyTorch 1.11   | - |
  | PyTorch 2.1   | - |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 下载Wikipedia单语数据，本模型训练时使用en，zh两种单语数据集，在源码包根目录下执行下载单语数据命令：
   ```
   ./get-data-wiki.sh en   # 下载英文单语数据
   ./get-data-wiki.sh zh   # 下载中文单语数据
   ```

2. 数据集目录结构参考如下所示。
   ```
   ├ data
   ├── processed
   │    ├── XLM_en_zh    
   │         ├── 50K
   │              ├── test.en.pth
   │              ├── test.zh.pth
   │              ├── train.en.pth
   │              ├── train.zh.pth
   │              ├── valid.en.pth
   │              ├── valid.zh.pth
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。


## 安装语言处理工具
方法一：
1. 进入源码包根目录下的tools路径。
   ```
   cd tools/
   ```
2. 安装摩西标记器。
   ```
   git clone https://github.com/moses-smt/mosesdecoder
   ```
3. 安装中文斯坦福分词器。
   ```
   wget https://nlp.stanford.edu/software/stanford-segmenter-2018-10-16.zip
   unzip stanford-segmenter-2018-10-16.zip
   ```
4. 安装fastBPE。
   ```
   git clone https://github.com/glample/fastBPE
   cd fastBPE
   g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
   ```

方法二：
1. 直接使用源码包根目录下的install-tools.sh脚本进行安装。

   ```
   ./install-tools.sh
   ```

2. 处理en和zh数据集，执行脚本tokenize_en_zh.sh。

   ```
   bash tokenize_en_zh.sh
   ```
执行结束后会在data/processed/XLM_en_zh/50k路径下生成处理好的en，zh数据集。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练（由于XLM模型在单卡训练时，loss不收敛，故不采用单卡精度训练）。

     ```
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```
     
   - 单机8卡训练
   
     启动8卡训练。
   
     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```
   
3. 指定单卡训练id。

   修改xlm/slurm.py脚本
   ```
   将168行，torch.npu.set_device(params.local_rank) 注释掉，并在其后添加如下一行
   torch.npu.set_device("npu:id")  # id可以设置为自己想指定的卡
   ```

--data_path参数填写数据集路径，需写到XLM_en_zh这一级。

模型训练脚本参数说明如下。

   ```
公共参数：
--seed                              //随机数种子设置
--fp16                              //采用半精度训练
--encoder_only                      //仅适用编码器
--use_memory                        //使用外部存储器
--data_path                         //数据集路径
--reload_checkpoint                 //重新加载权重文件
--exp_name                          //实验名称
--mem_enc_positions                 //编码器中的内存位置
   ```

训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME | AUC | FPS       | Epochs   |
| :------: | :------:  | :------: | :------: |
| NPU-1p | - | 160 | 1     |
| NPU-8p | 59.8 | 1087 | 180 |

   > **说明：** 
   > 8p的性能数据是在非二进制模式下获得的。

# 版本说明

## 变更

2023.03.08：更新readme，重新发布。

2021.07.08：首次发布。

## FAQ

XLM模型在8卡训练的编译阶段，使用的内存最大能达到315G左右，建议测试服务器要保证有大于320G的可用内存空间，才能拉起模型训练。