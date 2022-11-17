# MT5 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

mT5，即 Multilingual T5，是 T5 的多国语言版。 T5 模型是 Transfer Text-to-Text Transformer 的简写。跟BERT一样，T5也是Google出品的预训练模型。T5的理念就是“万事皆可Seq2Seq”，它使用了标准的Encoder-Decoder模型，并且构建了无监督/有监督的文本生成预训练任务，最终将效果推向了一个新高度。

- 参考实现：

  ```
  url=https://github.com/huggingface/transformers.git
  commit_id=61a51f5f23d7ce6b8acf61b5aa170e01d7658d74
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/nlp/
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套        | 版本                                                                            |
  |-------------------------------------------------------------------------------| ------------------------------------------------------------ |
  | 硬件 | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)                                                                    |
  | NPU固件与驱动   | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1)  |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)                        |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip3 install -r requirements.txt
  ```
- 配置运行环境。

  ```
  # 安装transformers
  cd transformers
  pip3 install -e .
  ```

## 准备数据集

1. 获取数据集。
   
   可以联网直接运行脚本，会自动下载数据集wmt16。

   也可以提前下载好数据集，放到 ./dataset 目录，然后软链接到指定路径，dataset目录结构如下

   ```
   ├── dataset
         ├── wmt16
             ├── datasets
                 ├── wmt16
                     ├── ro-en 
                         ├── 1.0.0
                            ├── 28ebdf...
                                ├── wmt16-train.arrow
                                ├── wmt16-validation.arrow
                                ├── wmt16-test.arrow
                                ├── dataset_info.json
             ├── modules
                ├── datasets_modules
                    ├── datasets
                    ├── metrics
   ```
   将wmt16下的内容软连接到/root/.cache/huggingface/下。
   ```
   ln -s $(pwd)/dataset/wmt16/datasets /root/.cache/huggingface/
   ln -s $(pwd)/dataset/wmt16/modules /root/.cache/huggingface/
   ```
   > **说明：** 
   >如果服务器可以联网，则不用搬移和软连接数据集，默认位置即可。

## 获取预训练模型

- 从https://huggingface.co/google/mt5-small/tree/main 中获取预训练模型放到指定的位置如./mt5-small中。需要下载的文件如下：config.json, pytorch_model.bin, special_tokens_map.json, spiece.model, tokenizer_config.json。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd ${模型文件夹名称}/
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --model_path=real_path
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh  --model_path=real_path
     ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
    --model_name_or_path  ./mt5-small  // 模型名称或路径
    --do_train \
    --do_eval \
    --source_lang en \        // 翻译的源语言
    --target_lang ro \      // 翻译的目标语言
    --source_prefix "translate English to Romanian: " \
    --dataset_name wmt16 \    
    --dataset_config_name ro-en \    
    --output_dir ./tst-translation \    # 模型输出目录
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --pad_to_max_length \
    --fp16 \
    --use_combine_grad True \
    --optim adamw_apex_fused_npu \
    --use_combine_ddp True \
    --half_precision_backend apex \
    --max_step 1000 \
    --save_step 5000
   ```


# 训练结果展示

**表 2**  训练结果展示表

| NAME   |    BLEU | 性能(it/s) |  steps | AMP_Type |
|--------|--------:|---------:|-------:|---------:|
| 1p-竞品A |       - |        - |   1000 |       O0 |
| 1p-NPU |       - |      4.5 |   1000 |       O2 |
| 8p-竞品A | 18.7866 |    24.32 |  57219 |       O0 |
| 8p-NPU | 18.3343 |    30.88 |  57219 |       O2 |


# 版本说明

## 变更

2022.11.14：首次发布

## 已知问题

无。











