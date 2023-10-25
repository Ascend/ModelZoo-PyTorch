# ChatGLM2-6B for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

ChatGLM**2**-6B 是开源中英双语对话模型 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，ChatGLM**2**-6B 引入了如下新特性：

1. **更强大的性能**：基于 ChatGLM 初代模型的开发经验，我们全面升级了 ChatGLM2-6B 的基座模型。ChatGLM2-6B 使用了 [GLM](https://github.com/THUDM/GLM) 的混合目标函数，经过了 1.4T 中英标识符的预训练与人类偏好对齐训练，[评测结果](https://github.com/THUDM/ChatGLM2-6B/tree/main#评测结果)显示，相比于初代模型，ChatGLM2-6B 在 MMLU（+23%）、CEval（+33%）、GSM8K（+571%） 、BBH（+60%）等数据集上的性能取得了大幅度的提升，在同尺寸开源模型中具有较强的竞争力。
2. **更长的上下文**：基于 [FlashAttention](https://github.com/HazyResearch/flash-attention) 技术，我们将基座模型的上下文长度（Context Length）由 ChatGLM-6B 的 2K 扩展到了 32K，并在对话阶段使用 8K 的上下文长度训练。对于更长的上下文，我们发布了 [ChatGLM2-6B-32K](https://huggingface.co/THUDM/chatglm2-6b-32k) 模型。[LongBench](https://github.com/THUDM/LongBench) 的测评结果表明，在等量级的开源模型中，ChatGLM2-6B-32K 有着较为明显的竞争优势。
3. **更高效的推理**：基于 [Multi-Query Attention](http://arxiv.org/abs/1911.02150) 技术，ChatGLM2-6B 有更高效的推理速度和更低的显存占用：在官方的模型实现下，推理速度相比初代提升了 42%，INT4 量化下，6G 显存支持的对话长度由 1K 提升到了 8K。
4. **更开放的协议**：ChatGLM2-6B 权重对学术研究**完全开放**，在填写[问卷](https://open.bigmodel.cn/mla/form)进行登记后**亦允许免费商业使用**。

- 参考实现：

  ```
  url=https://github.com/THUDM/ChatGLM2-6B
  commit_id=877ef10d85c93ddfcfe945fcdc764393a52541b8
  url=https://huggingface.co/THUDM/chatglm2-6b/tree/v1.0
  commit_id=0ade0d38ac00258ae09450696315c2ff0b1faf12
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

  | Torch_Version      |    三方库依赖版本     |
  |:--------------:| :----------------------------------------------------------: |
  | PyTorch 1.11 | deepspeed 0.9.2 |

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
  pip3 install deepspeed==0.9.2
  git clone https://gitee.com/ascend/DeepSpeed.git -b v0.9.2 deepspeed_npu
  cd deepspeed_npu
  git checkout 5c7c89930f0b70ea586d5db63f8e66477d5d9d9f
  pip3 install .
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

   用户从[链接](https://huggingface.co/THUDM/chatglm2-6b/tree/v1.0)自行获取模型文件（除了modeling_chatglm.py）和权重文件（pytorch_model-0000*-of-00007.bin
），并放于model目录下，微调依赖该模型权重。
   model参考目录如下
   
   ```
   ├── model
         ├──config.json
         ├──configuration_chatglm.py
         ├──ice_text.model
         ├──pytorch_model-00001-of-00007.bin
         ├──pytorch_model-00002-of-00007.bin
         ├──pytorch_model-00003-of-00007.bin
         ├──pytorch_model-00004-of-00007.bin
         ├──pytorch_model-00005-of-00007.bin
         ├──pytorch_model-00006-of-00007.bin
         ├──pytorch_model-00007-of-00007.bin
         ├──pytorch_model.bin.index.json
         ├──quantization.py
         ├──test_modeling_chatglm.py
         ├──tokenization_chatglm.py
         ├──tokenizer_config.json
   ```

- 替换modeling_chatglm.py文件

  ```
   
   # 使用fix文件夹下的modeling_chatglm.py替换到您下载的model目录下
  ```


## 

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
| Finetune -NPU |   (待补充)    |  (待补充)  | fp16  | 1.11  | 910 |
| Finetune -GPU |   (待补充)    |  (待补充)  | fp16  | 1.11  | A800 |

说明：P-Tuning 仅打通功能，无性能优化。

**表 2**  评估结果展示表

| 评估项  |   NPU   |   GPU   |
| :-----: | :-----: | :-----: |
| BLEU-4  | 8.0174  | 7.5779  |
| ROUGE-1 | 31.5737 | 31.0244 |
| ROUGE-2 | 7.2976  | 7.1179  |
| ROUGE-l | 24.8196 | 24.7112 |

说明：该结果是step=1000的验证结果。

# 版本说明

## 变更

2023.9.1：首次发布。

## FAQ

1. 报错提示deepspeed.py需要版本大于等于0.6.5
   ```
   # 关闭版本检测（如安装0.9.2版本无需此操作）
   # 若遇到该报错
   pip show transformers
   # 复制Location路径
   # 使用fix文件夹下的deepspeed.py替换路径下transformers/deepspeed.py
   ```

2. 加载参数阶段有卡死现象

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

