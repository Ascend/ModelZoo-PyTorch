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
  pip3 install -r requirements.txt
  ```


- 安装Ascend DeepSpeed 
  
  ```
  cd GPT-2_for_PyTorch
  pip3 install deepspeed==0.6.0  # 首先安装原生deepspeed
  git clone -b adaptor https://gitee.com/ascend/DeepSpeed.git
  cd ./DeepSpeed
  pip3 install ./
  cd -
  ```


- 克隆原始仓

  ```
  cd GPT-2_for_PyTorch
  git clone https://github.com/microsoft/Megatron-DeepSpeed.git
  cd ./Megatron-DeepSpeed
  git checkout b4d4a0e
  cd -
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


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd ./${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机8卡训练，由于多卡训练使能了多种并行特性，单卡没有相匹配的模型，故不支持单卡训练。

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=$real_data_path --model_size=$selected_model_size --train_iters=$train_iters
     bash ./test/train_performance_8p.sh --data_path=$real_data_path --model_size=$selected_model_size --train_iters=$train_iters
     ```
   - 备注1：model_size代表模型参数量，目前只提供了5种：345M、1.3B、2.7B、3.7B、345M_without_mp，用户可从5种之中选1种，也可不选默认model_size=345M
   - 备注2：train_iters代表训练迭代次数，不设置时默认值500000
   - 训练完成后，权重文件保存在./ckpts/ckpts_tmp下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME     | params |  PPL  | samples/s | seq_len | Steps     |
| -------  | -----  | -----  |------- | ----- | ------    |
| 8p-竞品V  | 345M  | 26.375  |   37.8 | 1024 | 500000    |
| 8p-NPU   | 345M | 26.485 |   59.7  | 1024 | 500000    |
| 8p-竞品V  | 1.3B  | - |  15.71 | 1024 |
| 8p-NPU   | 1.3B  | - |  21.47  | 1024 |
| 8p-竞品V  | 2.7B  | - |   4.125 | 2048 |
| 8p-NPU   | 2.7B  | - |   5.280  | 2048 |
| 8p-竞品V  | 3.7B  | - |   3.120 | 2048 |
| 8p-NPU   | 3.7B  | - |   4.558  | 2048 |

**表 3**  345M_without_mp训练结果展示表

| NAME     | params | PPL    | samples/s | seq_len | Steps     |
| -------  | -----  | -----  |------- | ----- | ------    |
| 8p-竞品A  | 345M  | 28.67 |   235 | 1024 | 100000    |
| 8p-NPU   | 345M  | 28.71 |   227  | 1024 | 100000    |

备注：一定要有竞品和NPU。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md

# 版本说明

## 变更

2023.03.20：首次发布
2023.05.24：首次变更

## FAQ

1. 针对Pytorch 2.0及以后版本，由于torch._six接口已弃用，且npu目前只支持0.6.0版本的deepspeed，可对应修改该三方库的源码文件。

   ```
   # 请参考以下路径修改源码文件，将文件中的 “from torch._six import inf” 修改为 “from math import inf”
   vim /usr/local/python3.8.10/lib/python3.8/site-packages/deepspeed/runtime/utils.py +18
   vim /usr/local/python3.8.10/lib/python3.8/site-packages/deepspeed/runtime/zero/stage_1_and_2.py +8
   vim /usr/local/python3.8.10/lib/python3.8/site-packages/deepspeed/runtime/zero/stage3.py +19
   vim /home/xxx/GPT-2_for_PyTorch/Megatron-DeepSpeed/megatron/optimizer/clip_grads.py +19
   ```