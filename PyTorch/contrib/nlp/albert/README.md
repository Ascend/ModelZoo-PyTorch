# Albert for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

Albert是自然语言处理模型，基于Bert模型修改得到。相比于Bert模型，Albert的参数量缩小了10倍，减小了模型大小，加快了训练速度。在相同的训练时间下，Albert模型的精度高于Bert模型。

- 参考实现：

  ```
  url=https://github.com/lonePatient/albert_pytorch
  branch=master 
  commit_id=46de9ec6b54f4901f78cf8c19696a16ad4f04dbc
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
  | 硬件       | [1.0.15.3](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)|
  | 固件与驱动 | [22.0.0.3](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC1.1](https://www.hiascend.com/software/cann/commercial?version=5.1.RC1.1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖：

  ```
  pip install -r requirements.txt
  ```



## 准备数据集

1. 获取数据集。

   下载[SST-2和STS-B](https://gitee.com/liuyj-suda-an/albert_full/tree/master/dataset)数据集,在模型根目录下创建dataset目录，并放入数据集。

   数据集目录结构参考如下所示。

   ```
   ├── dataset
         ├──SST-2
              ├──original
              │──dev.tsv
              │──test.tsv
              │──train.tsv   
              |  ...                     
         ├──STS-B  
              ├──original
              │──dev.tsv
              │──test.tsv
              │──train.tsv
              │   ...              
   ```

## 下载预训练模型
下载[albert_base_v2](https://gitee.com/liuyj-suda-an/albert_full/tree/master/prev_trained_model)，在模型根目录下创建prev_trained_model目录，并放入预训练模型。

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
   --data_dir                           //数据集路径
   --model_type                         //模型类型
   --task_name                          //任务名称
   --output_dir                         //输出保存路径
   --do_train                           //是否训练
   --do_eval                            //是否验证
   --num_train_epochs                   //重复训练次数
   --batch-size                         //训练批次大小
   --learning_rate                      //初始学习率
   --fp16                               //是否使用混合精度
   --fp16_opt_level                     //混合精度的level
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。



# 训练结果展示

**表2**  训练结果展示表

| NAME    | Acc@1 |  FPS | Epochs | 
| :-----: | :---: | :--: | :----: | 
| 1p-竞品 | 0.927 |  517 |   2   |
| 1p-NPU  | 0.932 |  393 | 2     |
| 8p-竞品 | 0.914 | 3327 | 7    |
| 8p-NPU  | 0.927 | 2675 | 7    |



# 版本说明

## 变更

2022.08.24：首次发布



## 已知问题

无。











