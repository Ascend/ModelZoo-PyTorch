# GPT-2_for_PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

Megatron 和 DeepSpeed 是两个很重要的预训练框架。Megatron 是英伟达做的超大规模预训练模型框架，主要是利用 tensor parallel 做性能优化以及 mode parallel。DeepSpeed 是微软团队做的深度学习加速框架。这两个团队合作构造出 Megatron-DeepSpeed  框架，相当于是把两个框架的特点结合在一起，并用它训练一个 530B 的模型。

- 参考实现：

  ```
  url=https://github.com/microsoft/Megatron-DeepSpeed.git
  commit_id=b4d4a0e
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/nlp
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.8 | - |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r requirements.txt  # PyTorch1.8版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。

- 安装Ascend DeepSpeed 
  
  请参考并完成安装（https://gitee.com/ascend/DeepSpeed.git）

- 克隆原始仓

  ```
  cd GPT-2_for_PyTorch
  git clone https://github.com/microsoft/Megatron-DeepSpeed.git
  # 进入github上拉下来的Megatron-DeepSpeed
  cd ./Megatron-DeepSpeed
  git checkout b4d4a0e
  cd -
  ```

- 安装依赖（根据模型需求，按需添加所需依赖）。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

    ```bash ./test/dataset_preprocess_gpt.sh```

2. 数据集目录结构
   将数据集默认放置在```./data/```下，数据集的目录结构如下所示：

   ```
   ├── ./data/
         ├── gpt2-vocab.json         
         ├── gpt2-merges.txt
         ├── my-gpt_text_sentence.bin
         ├── my-gpt_text_sentence.idx
   ```

> **说明：** 
>该数据集的训练过程脚本只作为一种参考示例。


## 获取预训练模型（可选）

- 本模型不涉及


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd ./${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机8卡训练。

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./tests/train_GPT345M_full_8p.sh   
     ```
   
   训练完成后，权重文件保存在./ckpts/ckpts_tmp下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表(GPT-2-345M)

| NAME     | PPL    | samples/s | Steps     |
| -------  | -----  |----------:| ------    |
| 8p-竞品V  | 2.668  |   37.8 | 500000    |
| 8p-NPU   | 2.676 |   59.7  | 500000    |

备注：一定要有竞品和NPU。

# 版本说明

## 变更

2023.03.20：首次发布

## 已知问题

**_当前发行版本中存在的问题描述。_**

无。