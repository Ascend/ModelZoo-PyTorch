# Speech-Transformer for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

Speech-Transformer是使用Transformer网络实现的一个端到端的自动语音识别网络，它能够将声音特征直接转化成文字。

- 参考实现：

  ```
  url=https://github.com/kaituoxu/Speech-Transformer
  commit_id=e6847772d6a786336e117a03c48c62ecbf3016f6
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/audio
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  在模型源码包根目录下执行命令。
  ```
  pip3.7 install -r requirements.txt
  ```

## 准备数据集

1. 获取数据集。

   用户自行下载 `data_aishell` 数据集，上传到服务器模型源码包根目录下新建的 `utils` 文件夹下并解压。解压后进入到 `data_aishell/wav` 文件夹下解压所有的压缩包，命令如下：
   ```
   ls *.tar.gz | xargs -n1 tar xzvf
   ```
   在 `utils/` 文件夹下安装 `kaldi`，请参考[INSTALL](https://github.com/kaldi-asr/kaldi)进行安装。
    
   数据集目录结构参考如下所示。
   ```
   ├── utils
        ├──data_aishell
             ├──wav
                 ├──train
                      │──S0002
                      │──S0003
                      ├──...  
                      ├──S0111    
                  ├──test
                      │──S0764
                      │──S0765
                      ├──S0766 
                      ├──S0767  
                      │──S0768
                      │──S0769
                      ├──S0770 
                      ├──S0901 
                      │──S0902
                      ├──...
                      ├──S0916 
        ├──kaldi         
   ``` 
   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理。
   
   提取声音特征。
   ```
   cd ${模型文件夹名称}/tools         # 进入到源码包根目录下的tools目录
   make KALDI=/XXX/utils/kaldi       # XXX为utils文件夹中kaldi安装的源码位置
   cd ../test                        # 进入到源码包根目录下的test目录，修改init.sh中data变量指向数据集data_aishell的上一层目录
   bash init.sh                      # 在test目录下执行，提取声音特征 
   ```

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录下的test目录。

     ```
     cd ${模型文件夹名称}/test 
     ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash train_full_1p.sh  # 单卡精度
     
     bash train_performance_1p.sh # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash train_full_8p.sh  # 8卡精度
     
     bash train_performance_8p.sh # 8卡性能
     ```

   - 单机8卡评测。

     启动8卡评测。
     ```
     bash train_eval_8p.sh  # 启动评测脚本前，需对应修改评测脚本中的--model-path参数，指定ckpt文件路径
     ```
   
   --model-path参数为训练权重生成路径，需写到权重文件的一级目录。

   模型训练脚本参数说明如下。
   ```
   公共参数：
   --train-json                        //训练数据集路径
   --valid-json                        //验证数据集路径
   --dict                              //训练集字典
   --batch-size                        //训练批次大
   --k                                 //优化器参数
   --warmup_steps                      //优化器参数
   --label_smoothing                   //loss计算参数
   --epochs                            //训练epochs
   --contine_from                      //是否加载预训练权重           
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | CER | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V | - | - | 1 | - | 1.5 |
| 8p-竞品V | - | - | 150 | - | 1.5 |
| 1p-NPU | -        | 178.4    | 1        | O2       | 1.8 |
| 8p-NPU | 10.0     | 1301.5   | 150      | O2       | 1.8 |

# 版本说明

## 变更

2023.1.10：更新readme，重新发布。

## FAQ

无。

