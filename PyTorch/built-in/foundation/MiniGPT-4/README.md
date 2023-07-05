# MiniGPT-4 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

MiniGPT-4使用一个投影层将来自BLIP-2的冻结视觉编码器与冻结的LLM Vicuna对齐。通过两个阶段来训练MiniGPT-4，先是用500万图文对训练，然后再用一个3500对高质量数据集训练。

- 参考实现：

  ```
  url=https://github.com/Vision-CAIR/MiniGPT-4
  commit_id=22d8888ca2cf0aac862f537e7d22ef5830036808
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/foundation
  ```

- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、CANN以及PyTorch如下表所示。

  **表 1**  版本支持表

  | 配套       | 版本                                 |
  | :--------: | :------------: |
  | 固件       | 6.4.t8.0.b207 |
  | 驱动       | 23.0.rc2.b031 |
  | CANN       | 6.3.T200 |
  | PyTorch    | [1.11.0](https://gitee.com/ascend/pytorch/tree/v1.11.0/) |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。

  ```
  pip install -r requirements.txt  # PyTorch1.11版本
  ```



## 准备数据集

1. 获取预训练数据集。

   要下载和准备Laion和CC数据集，请查看[第一阶段数据集准备说明](dataset/README_1_STAGE.md)。
   数据集参考目录如下
   ```
   laion_dataset
   ├── 00000.parquet
   ├── 00000_stats.json
   ├── 00000.tar
   ├── ...
   
   cc_sbu_dataset
   ├── 00000.parquet
   ├── 00000_stats.json
   ├── 00000.tar
   ├── ...
   ```

2. 获取微调数据集

   要下载和准备小型高质量图像文本对数据集，请查看[第二阶段数据集准备说明](dataset/README_2_STAGE.md)。
   数据集参考目录如下
   ```
   cc_sbu_align
   ├── filter_cap.json
   ├── image
      ├── 0.jpg
      ├── ...
   
   ```

## 准备模型权重

1. 准备预训练的Vicuna权重

   用户参照[链接](PrepareVicuna.md)自行获取模型文件，并放于model目录下，微调依赖该模型权重。
   model参考目录如下
   ```
   vicuna_weights
   ├── config.json
   ├── generation_config.json
   ├── pytorch_model.bin.index.json
   ├── pytorch_model-00001-of-00003.bin
   ```
  
2. 准备训练的MiniGPT-4检查点

   | Checkpoint Aligned with Vicuna 3B |  Checkpoint Aligned with Vicuna 7B  |
   :-------------:|:-------------:
   [链接](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link) | [链接](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) 

   然后，在评估配置文件[minigpt4_eval.yaml](eval_configs/minigpt4_eval.yaml#L10)的第11行中设置预训练检查点的路径。

3. 准备只有第一阶段训练的MiniGPT-4检查点[链接](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link)

# 开始训练

## 预训练模型


```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```



## 第二阶段微调

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

# 训练结果展示

**表 1**  预训练结果展示表


|     NAME      | TokensPerSec | Iterations  | BatchSize  | Torch_Version | 
|:-------------:|:-------------:|:-:|:-:|:-:|
| Pretrain -GPU |     8866      | 5000*4   | 64  | 1.11  | 
| Pretrain -NPU |     7517      | 8000*4   | 40  | 1.11  | 


**表 2**  微调结果展示表
|     NAME      | TokensPerSec | Iterations  | BatchSize  | Torch_Version | 
|:-------------:|:-------------:|:-:|:-:|:-:|
| Finetune -GPU |     2805      | 200*2   | 12  | 1.11  | 
| Finetune -NPU |     2433      | 240*2   | 10  | 1.11  | 

# 版本说明

## 变更

2023.7.05：首次发布。

## FAQ

无