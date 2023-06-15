# LayoutlMv2 for PyTorch

-   [概述](#1)
-   [准备训练环境](#2)
-   [开始训练](#3)
-   [训练结果展示](#4)
-   [版本说明](#5)

# 概述

## 简述

LayoutLMv2 是 LayoutLM 的改进版本，具有新的预训练任务，可在单个多模态框架中对文本、布局和图像之间的交互进行建模。它的性能优于强大的基线，并在各种下游视觉丰富的文档理解任务上取得了最先进的新结果，包括 FUNSD （0.7895 → 0.8420）、CORD （0.9493 → 0.9601）、SROIE （0.9524 → 0.9781）、Kleister-NDA （0.834 → 0.852）、RVL-CDIP （0.9443 → 0.9564） 和 DocVQA （0.7295 → 0.8672）。

- 参考实现：
  
  ```bash
    url=https://github.com/microsoft/unilm/tree/master/layoutxlm
    commit_id=ec8c2624c8832aa4ca89005fd20e85a211f20a8f
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

  | Torch_Version     | 三方库依赖版本 
  | --------          |:---------:
  | PyTorch 1.8       | transformers==4.5.1; detectron2==0.3; seqeval==1.2.2; datasets==2.7.1; packaging==21.0

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。


  ```bash
  # 安装detectron2
  git clone https://github.com/facebookresearch/detectron2.git -b v0.3
  python -m pip install -e detectron2
  ```

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。

  ```bash
  pip install -r requirements.txt
  pip install -e .
  ```
  
## 准备数据集
- 在有网络的情况下，模型训练需要的数据集会在训练开始之前由训练脚本自动下载，无需准备数据集。

- 在没有网络的情况下，用户也可以自行下载funsd数据集，并且移动编译到 */root/.cache/huggingface/datasets/funsd/funsd/1.0.0/4a44e695553877e8dfc6213fe7a8974940ca95a07dbd543bd77a8f7da6c4e5a3* 路径下，数据集目录参考结构如下所示：

   ```
   1.0.0
   |——————4a44e695553877e8dfc6213fe7a8974940ca95a07dbd543bd77a8f7da6c4e5a3
   |        └——————dataset_info.json
   |        └——————funsd-test.arrow
   |        └——————funsd-train.arrow   
   ```
## 获取预训练模型
本文使用layoutlmv2-base-uncased预训练模型 

- 用户在有网络的情况下，预训练模型会在训练开始之前由训练脚本自动下载。

- 在没有网络的情况下，需要用户自行下载预训练模型layoutlmv2-base-uncased，将获取的预训练模型上传至 */root/.cache/huggingface/transformers/3f564f098a94cd2c8fcf76cf65649d4f0482b917cb8552b34e266a0e5d7808a7.f7616351b25aa8250bddb46d3adb2faa9f5f7209ec75f3b4cee8d1dd2c7ca920* 目录下。


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
     bash test/train_full_1p_v2.sh   # 任务
     bash test/train_performance_1p_v2.sh   # 性能任务
       
     ```
   
   + 单机8卡训练
   
     启动8卡训练：
   
     ```bash
     bash test/train_full_8p_v2.sh   # 任务
     bash test/train_performance_8p_v2.sh   # 性能任务
       
     ```
     
     
   + 脚本中调用的python命令参数说明如下：
     
      ```bash
      --output_dir                                   // 训练结果和checkpoint保存路径
      --nproc_per_node                               // 训练使用的卡数
      --model_name_or_path                           // 预训练模型文件夹路径
      --do_train                                     // 执行训练
      --do_predict                                   // 执行预测
      --fp16                                         // 使用混合精度
      --warmup_ratio                                 // warmup率，用于调整学习率
     ```
     
     训练完成后，权重文件保存在output_dir路径下，并输出模型训练精度和性能信息。
     
     

# 训练结果展示

**表 2**  训练结果展示表

***test***

| NAME     | test f1 |   FPS    | AMP_Type | Epochs | Batch Size |
| -------- |:---------:|:--------:| :------: | ------ | ---------- |
| 1p-NPU   |    -     |  7.002   |    O1    | 52.63  | 8         |
| 1p-竞品V |    -     |  15.784  |    O1    | 52.63   | 8         |
| 1p-竞品A |    -     |  19.488  |    O1    | 52.63   | 8         |
| 8p-NPU   |  0.8213   | 55.325   |    O1    | 333.33 | 64         |
| 8p-竞品V |  0.8184   | 71.616   |    O1    | 333.33  | 64         |
| 8p-竞品A |  0.822   | 158.272   |    O1    | 333.33  | 64         |


# 版本说明

## 变更

2023.05.04：内测版本
## FAQ
1.下载数据集时，出现报错**SSLCertVerificationError**时，可以将 _/site-packages/requests/api.py_ 下的 
```python 
return session.request(method=method, url=url, **kwargs)  
```
修改为
```python 
return session.request(method=method, url=url, verify=False, **kwargs)
```

2.Tokenizer初始化失败，需要Rust.
运行：
```python 
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```