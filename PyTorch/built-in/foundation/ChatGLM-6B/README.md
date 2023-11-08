# Chat-GLM-6B for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

ChatGLM-6B 是一个开源的、支持**中英双语**的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数。

本仓支持**P-Tuning v2 和全参数fintune**。

- 参考实现：

  ```
  url=https://github.com/THUDM/ChatGLM-6B
  commit_id=27b04bce90b34e719375576cc67ff5374bb2f38a
  url=https://huggingface.co/THUDM/chatglm-6b/tree/v0.1.0
  commit_id=f83182484538e663a03d3f73647f10f89878f438
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/foundation
  ```


# 准备训练环境

## 准备环境

默认配置需要每张卡有60G以上空闲内存。

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | 软件名称  |    版本    |                             链接                             |
  |:--------------:| :----------------------------------------------------------: |:--------------:|
  | CANN | 7.0.RC1 | [LINK](https://support.huawei.com/carrier/productNewOffering?col=product&path=PBI1-21430725/PBI1-21430756/PBI1-22892969/PBI1-23710427/PBI1-251168373&resTab=SW) |
  | Atlas 800T A2 | 1.0.RC3 | [LINK](https://support.huawei.com/carrier/productNewOffering?col=product&path=PBI1-21430725/PBI1-21430756/PBI1-22892969/PBI1-23710427/PBI1-254184887) |
  | FrameworkPTAdapter | 5.0.RC3 |      [LINK](https://gitee.com/ascend/pytorch/releases)       |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  1. 安装基础依赖

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r requirements.txt  # PyTorch1.11版本
  ```

  2. 安装deepspeed_npu插件

  ```
  # v0.9.2分支
  git clone https://gitee.com/ascend/DeepSpeed.git
  cd Deepspeed
  pip3 install ./
  ```
  
  3. 替换transformers依赖文件
  ```
   # 自动替换无法替换三方库中的文件。
   pip show transformers
   # 获取transformers的Location路径
   # 使用fix文件夹下的tranining_args.py替换路径下transformers/tranining_args.py
  ```


## 准备数据集

1. 获取数据集。

   ADGEN 数据集任务为根据输入（content）生成一段广告词（summary）。

   从 [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing) 或者 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) 下载处理好的 ADGEN 数据集，将解压后的 `AdvertiseGen` 目录放到ptuning目录下。
   数据集参考目录如下
   ```
   ├── AdvertiseGen
         ├──train.json
         ├──dev.json
   ```
2. 预处理数据集。
为了方便启动训练后，不用再每次重复加载处理数据集，故提前进行处理。也可以下载提前处理好的[数据集](https://pan.baidu.com/s/1hA9yfqJkKi1Ae5FB-AkJcQq)，提取码7s5i
```shell
bash preprocess.sh
```
处理好的数据集位于同目录下的train_datasets文件夹下，参考目录如下
```
   ├── train_datasets
         ├──data-00000-of-00008.arrow
         ├──data-00001-of-00008.arrow
         ├──data-00002-of-00008.arrow
         ├──data-00003-of-00008.arrow
         ├──data-00004-of-00008.arrow
         ├──data-00005-of-00008.arrow
         ├──data-00006-of-00008.arrow
         ├──data-00007-of-00008.arrow
         ├──dataset_info.json
         ├──state.json
```


## 准备模型权重

1. 获取语言识别模型和预训练权重

   用户从[链接](https://huggingface.co/THUDM/chatglm-6b/tree/v0.1.0)自行获取模型文件（除了modeling_chatglm.py）和8份权重文件（pytorch_model-0000*-of-00008.bin
   ），并放于model目录下，微调依赖该模型权重。
   model参考目录如下
   ```
   ├── model
         ├──config.json
         ├──configuration_chatglm.py
         ├──ice_text.model
         ├──pytorch_model-00001-of-00008.bin
         ├──pytorch_model-00002-of-00008.bin
         ├──pytorch_model-00003-of-00008.bin
         ├──pytorch_model-00004-of-00008.bin
         ├──pytorch_model-00005-of-00008.bin
         ├──pytorch_model-00006-of-00008.bin
         ├──pytorch_model-00007-of-00008.bin
         ├──pytorch_model-00008-of-00008.bin
         ├──pytorch_model.bin.index.json
         ├──quantization.py
         ├──test_modeling_chatglm.py
         ├──tokenization_chatglm.py
         ├──tokenizer_config.json
   ```



# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}/ptuning
   ```
   修改ptuning目录下的env_npu.sh，修改引用的环境变量位置。

2. 运行训练脚本。

   该模型P-Tuning v2支持单机单卡，全参数fintune支持单机8卡。

   - P-Tuning v2

     启动P-Tuning v2。

     ```
     bash train.sh
     ```

   - 全参数finetune

     启动8卡微调。
     可以用deepspeed.json配置deepspeed参数，目前默认使用zero2

     ```
     bash ds_train_fintune.sh 
     ```

   
   模型训练参数说明如下。

   ```
   公共参数：
   --max_source_length                       //处理后句子长度
   --max_target_length                       //目标数据长度
   --per_device_train_batch_size             //每卡训练批次大小
   --gradient_accumulation_steps             //梯度更新步数
   --max_steps                               //最大训练步数
   --logging_steps                           //打印信息步数
   --save_steps                              //保存参数步数
   --learning_rate                           //学习率
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练相关信息。

## 验证模型

1. 全参数finetune验证

    运行以下命令
    ```
    cd /${模型文件夹名称}/ptuning
    bash evaluate_fintune.sh 
    ```
    生成结果在屏幕上显示

# 训练结果展示

**表 1**  训练结果展示表


|     NAME      | SamplesPerSec | Iterations  | DataType  | Torch_Version | Card |
|:-------------:|:-------------:|:-:|:-:|:-:|:----:|
| Finetune -NPU |     2213      | 5000   | fp16  | 1.11  | 910 |
| Finetune -GPU |     2048      | 5000   | fp16  | 1.11  | A800 |

说明：P-Tuning 仅打通功能，无性能优化。

**表 2**  评估结果展示表

|   评估项   |   NPU   |   GPU   | 
|:-------:|:-------:|:-------:|
| BLEU-4  | 8.2853  | 8.1127  |
| ROUGE-1 | 31.1898 | 30.7429 |
| ROUGE-2 | 7.3583  | 7.1024  |
| ROUGE-l | 24.9874 | 24.8157 |

说明：该结果是step=5000的验证结果。

# 版本说明

## 变更

2023.6.25：首次发布。

## FAQ

1. 报错提示deepspeed.py需要版本大于等于0.6.5
   ```
   # 关闭版本检测（如安装0.9.2版本无需此操作）
   # 若遇到该报错
   pip show transformers
   # 复制Location路径
   # 使用fix文件夹下的deepspeed.py替换路径下transformers/deepspeed.py
   ```

2. 报错checkpoint.py

   ```
   # 1.11版本适配问题，新版本已修复。
   # 若遇到该报错
   pip show torch_npu
   # 复制Location路径
   # 使用fix文件夹下的checkpoint.py替换路径下torch_npu/utils/checkpoint.py
   
   ```
3. 加载参数阶段有卡死现象
   
   ```
   删除root下的cache目录，重新运行
   ```
4. 单卡阶段报embedding_dense_grad算子错误
   ```
   enbedding当前版本，不支持动静合一，静态有部分shape不支持,新版本已修复
   # 若遇到该报错
   修改main.py文件
   torch.npu.set_compile_mode(jit_compile=False)
   ```
5. 提示so文件错误
   ``` 
   提示so文件找不到
   # 若遇到该报错
   全局搜索so的位置，然后导入环境变量
   export LD_LIBRARY_PATH=/usr/:$LD_LIBRARY_PATH
   ```
6. eval提示scaledsoftmax报错
    ``` 
   算子shape泛化性还有问题
   # 若遇到该报错
   搜索output文件夹生成的modeling_chatglm.py文件，
   self.scale_mask_softmax 设置为false
   ```