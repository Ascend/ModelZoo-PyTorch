# RoBERTa for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

RoBERTa 模型更多的是基于 BERT 的一种改进版本。是 BERT 在多个层面上的重大改进。
RoBERTa 在模型规模、算力和数据上，都比 BERT 有一定的提升。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/fairseq/tree/main/examples/roberta 
  commit_id=d871f6169f8185837d1c11fb28da56abfd83841c
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/nlp
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

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 硬件       | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)  |
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖：

  ```
  pip install -r requirements.txt
  ```



## 准备数据集

1. 获取数据集。

   下载`SST-2`数据集，请参考`examples/roberta/preprocess_GLUE_tasks.sh`。

   `SST-2`数据集目录结构参考如下所示。

   ```
   ├── SST-2
         ├──input0
              ├──dict.txt
              │──preprocess.log
              │──test.bin
              │——test.idx   
              ├──train.bin
              │──train.idx
              │──valid.bin
              │——valid.idx                    
         ├──lable
              ├──dict.txt
              │──preprocess.log 
              ├──train.bin
              │──train.idx
              │──valid.bin
              │——valid.idx              
   ```



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
     bash ./test/train_full_1p.sh --data_path=real_data_path   
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=real_data_path  
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                              //数据集路径
   --restore-file                           //权重文件保存路径
   --max-tokens                             //最大token值
   --num-classes                            //分类数      
   --max-epoch                              //重复训练次数
   --batch-size                             //训练批次大小
   --lr                                     //初始学习率，默认：0.01
   --use-apex                               //使用混合精度
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。



# 训练结果展示

**表2**  训练结果展示表

| NAME    | Acc@1 |  FPS | Epochs |
| :-----: | :---: | :--: | :----: |
| 1p-竞品 | 0.927     |  397 |   1   |
| 1p-NPU  | 0.938     |  216 | 1     |
| 8p-竞品 | 0.943 | 2997 | 10    |
| 8p-NPU  | 0.969 | 1985 | 10    |



# 版本说明

## 变更

2022.08.24：首次发布



## 已知问题

无。











