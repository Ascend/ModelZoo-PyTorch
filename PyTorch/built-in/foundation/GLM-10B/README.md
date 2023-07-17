# GLM-10B for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

GLM是一个用自回归完型填空目标预训练的通用语言模型，可以在各种自然语言理解和生成任务中进行微调。


- 参考实现：

  ```
  url=https://github.com/THUDM/GLM
  commit_id=4f61ed7237a3b0187f4d62062429348276a78c84
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

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.11 | deepspeed 0.6.0 |
  
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
  git clone https://gitee.com/ascend/DeepSpeed.git
  cd Deepspeed
  pip3 install ./
  ```

  3. 安装pdsh

  获取[pdsh-2.34](https://github.com/chaos/pdsh/releases/tag/pdsh-2.34)源码并解压。
  ```
  cd pdsh-2.34
  ./configure --with-ssh
  make && make install
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，预训练可选用的开源数据集The Pile等，微调可选用COPA数据集等。将数据集上传到服务器任意路径下并解压。

   GLM-10B预训练使用到的Pile数据集目录结构参考如下所示，完整数据集解压处理后近3T，仅使用部分数据00.jsonl预训练作为参考。多机预训练时，每节点上均须将Pile数据集移动或软连接到模型脚本目录下。默认仅使用00.jsonl文件进行预训练。
   
   目录结构参考如下：

   ```
   ├── GLM-10B
        ├── pile
             ├──00.jsonl
             ├──02.jsonl
             ├   ...
             ├──29.jsonl
   ```

   微调可使用COPA数据集，目录结构参考如下。

   ```
   ├── COPA
         ├──train.jsonl
         ├──test.jsonl
         ├──val.jsonl
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

2. 获取词表

   用户自行获取gpt2词表，并放在模型目录中.pytorch_pretrained_bert目录下，该目录可从参考实现链接中获取，目录结构如下。
   ```
   ├── .pytorch_pretrained_bert
         ├──gpt2-merges.txt
         ├──gpt2-vocab.json
   ```

## 准备模型权重

1. 获取语言识别模型

   用户自行获取语言识别模型lid.176.bin，并放于模型目录下，预训练或微调均依赖该模型权重。

2. 获取预训练权重

   用户自行获取预训练权重，如glm-10b-1024，可从参考实现链接中获取，在模型目录下创建checkpoints目录，并将预训练权重放入其中。微调依赖该权重。目录结构如下。
   ```
   ├── checkpoints
         ├──glm-10b-1024
             ├──126000
                 ├──mp_rank_00_model_states.pt
             ├──latest
             ├──latest_checkpointed_iteration.txt
   ```

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   该模型预训练支持双机16卡，微调支持单机8卡。

   - 双机16卡预训练
     修改`hostfile`文件，参考默认配置修改为双机IP。

     启动16卡训练。

     ```
     bash ./tests/train_pretrain_full_16p.sh # 16卡预训练长稳
     
     bash ./tests/train_pretrain_performance_16p.sh # 16卡预训练性能
     ```

     --data_path参数填写数据集路径，若仅使用一个jsonl文件，指定到具体的文件，若使用多个，指定到上一级目录；

   - 单机8卡微调

     启动8卡微调。

     ```
     bash ./tests/train_finetune_full_8p.sh --data_path=/data/xxx/  # 8卡微调精度，
     ```
     --data_path参数填写数据集路径，需写到微调数据集的上一级目录，如COPA的上一级；



   模型训练参数说明如下。

   ```
   公共参数：
   --train-iters                       //训练总迭代数
   --fp16                              //是否使用fp16训练
   --train_micro_batch_size_per_gpu    //每卡训练批次大小
   --lr                                //学习率
   --stage                             //ZeRO stage配置
   --seed                              //使用随机数种子
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。
   

# 训练结果展示

**表 2**  训练结果展示表

| NAME  | SamplesPerSec  | Iterations  | DataType  | Torch_Version |
|:-:|:-:|:-:|:-:|:-:|
| Pretrain 16p-NPU  | 25  | 5000   | fp16  | 1.11  |
| Pretrain 16p-GPU  | 31  | 5000   | fp16  | 1.11  |

| NAME  | Accuracy  | Epochs  | DataType  | Torch_Version |
|:-:|:-:|:-:|:-:|:-:|
| Finetune 8p-NPU  | 98  | 100 | fp16  | 1.11  |
| Finetune 8p-GPU  | 98  | 100 | fp16  | 1.11  |

   > **说明：** 
   > Accuracy指微调过程中最高精度，实际验证NPU与GPU均存在1~2%波动。

# 版本说明

## 变更

2023.6.10：首次发布。

## FAQ

1. 报错torch.cuda.nvtx缺少range属性

   ```
   # 早期版本适配问题，新版本已修复。
   # 若遇到该报错，修改deepspeed/utils/nvtx.py如下。
   
   def instrument_w_nvtx(func):
       return func
   
   ```
   
# 公网地址说明

代码涉及公网地址参考 public_address_statement.md