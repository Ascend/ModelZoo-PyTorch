# MT5 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

MT5，即 Multilingual T5，是 T5 的多国语言版。 T5 模型是 Transfer Text-to-Text Transformer 的简写。跟BERT一样，T5也是Google出品的预训练模型。T5的理念就是“万事皆可Seq2Seq”，它使用了标准的Encoder-Decoder模型，并且构建了无监督/有监督的文本生成预训练任务，最终将效果推向了一个新高度。

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

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |
  | PyTorch 2.1 | scikit-learn==1.2.2 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  ```
  pip install -r requirements.txt
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

   也可以提前下载好数据集，放到新建立的 ./dataset 目录中，然后软链接到指定路径，dataset目录结构如下。

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

- 用户自行获取预训练模型并放到指定的位置，如存放到新建立的./mt5-small目录中。需要下载的文件如下：config.json， pytorch_model.bin，special_tokens_map.json，spiece.model，tokenizer_config.json。


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
     bash ./test/train_full_1p.sh --model_path=real_path  # 单卡精度
     
     bash ./test/train_performance_1p.sh --model_path=real_path  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --model_path=real_path  # 8卡精度
     
     bash ./test/train_performance_8p.sh --model_path=real_path  # 8卡性能
     ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh --model_path=real_path
     ```

   --model_path参数填写预训练模型路径，写到一级目录即可。
   
   模型训练脚本参数说明如下。

   ```
   公共参数：
   --addr                              //主机地址
   --weight_decay                      //权重衰减
   --seed                              //随机数种子设置
   --npu_id                            //npu训练卡id号
   --loss_scale                        //混合精度loss scale大小
   --init_checkpoint                   //模型权重初始化
   --train_batch_size                  //训练批次大小
   --use_npu                           //使用npu进行训练
   --num_train_epochs                  //训练周期数
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   |  BLEU   | FPS(it/s) | steps | AMP_Type | Torch_Version |
| :------: | :-----: | :-------: | :---: | :------: | :-----------: |
| 1p-竞品A |    -    |     -     | 1000  |    O0    |      1.5      |
| 8p-竞品A | 18.7866 |   24.32   | 57219 |    O0    |      1.5      |
|  1p-NPU-非ARM  |    -    |    4.5    | 1000  |    O1    |      1.8      |
|  8p-NPU-非ARM  | 18.3343 |   30.88   | 57219 |    O1    |      1.8      |
|  8p-NPU-ARM  |    -    |   24.96   | 57219 |    O1    |      1.8      |


# 版本说明

## 变更

2023.04.27：更新readme中8卡arm环境性能基线，重新发布。

2023.02.20：更新readme，重新发布。

2022.11.14：首次发布。

## FAQ

无。	

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md