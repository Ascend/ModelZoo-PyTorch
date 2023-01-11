# Speech-Transformer_for_PyTorch

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
  | 硬件       | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```
  pip3.7 install -r requirements.txt
  ```

## 准备数据集


1. 获取数据集。

   - 在模型目录下创建文件夹"utils/"，新建命令如下：
    
    ```
    mkdir utils
    ```

   - 进入"utils/"文件夹，下载data_aishell数据集并解压，命令如下：
    
    ```
    wget https://openslr.magicdatatech.com/resources/33/data_aishell.tgz
    tar xvf aidata_shell.tgz
    ```
   
   - 进入到“data_aishell/wav”文件夹里，解压里面的所有的压缩包，命令如下：
    
    ```
     ls *.tar.gz | xargs -n1 tar xzvf
    ```
   
   - 在"utils/"文件夹中安装kaldi，请参考[INSTALL](https://github.com/kaldi-asr/kaldi)进行安装。
    
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
    cd ${模型文件夹名称}/tools                //进入到源码包根目录下的tools目录
    make KALDI=/XXX/utils/kaldi            //XXX为utils文件夹中kaldi安装的源码位置
    cd ../test                             //进入到源码包根目录下的test目录,修改init.sh中data变量指向数据集data_aishell的上一层目录
    bash init.sh                           //在test目录下执行，提取声音特征 
    ```

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

     ```
     cd ${模型文件夹名称}/test 
     ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash train_full_1p.sh 
     
     bash train_performance_1p.sh 
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash train_full_8p.sh 
     
     bash train_performance_8p.sh 
     ```

   - 单机8卡评估。
     ```
     bash train_eval_8p.sh
     ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --batch_size                        //训练批次大
   --k                                 //优化器参数
   --warmup_steps                      //优化器参数
   --label_smoothing                   //loss计算参数
   --epochs                            //训练epochs
   --contine_from                      //是否加载预训练权重           
   ```

# 训练结果展示

**表 2**  训练结果展示表

| NAME | CER    | Npu_nums | Epochs   | AMP_Type | FPS | 
| :------: | :------: | :------: | :------: | :------: | :------: |
| 1.8 | -        | 1        | 150      | O2       | 178.4 |
| 1.5 | -        | 1        | 150      | O2       | 130 |
| 1.8 | 10.0     | 8        | 150      | O2       | 1301.5 |
| 1.5 | 9.9      | 8        | 150      | O2       | 855 |
# 版本说明

## 变更

2023.1.10：更新readme，重新发布。


## 已知问题

暂无。

