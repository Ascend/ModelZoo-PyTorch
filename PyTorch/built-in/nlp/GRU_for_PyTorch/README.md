# GRU for PyTorch

-   [概述](#gaishu)
-   [准备训练环境](#huanjing)
-   [开始训练](#xunlian)
-   [训练结果展示](#jieguo)
-   [版本说明](版本说明.md)



# <span id="gaishu">概述</span>

## 简述

GRU（Gate Recurrent Unit）是循环神经网络（Recurrent Neural Network, RNN）的一种。 和LSTM（Long-Short Term Memory）一样，也是为了解决长期记忆和反向传播中的梯度等问题而提出来的。

- 参考实现：

  ```
  url=https://github.com/farizrahman4u/seq2seq
  commit_id=c37c67ffccc7578d03dd97100dffd99cc675c85d
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/nlp
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# <span id="huanjing">准备训练环境</span>

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)|

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

    1.安装spacy。 
    ```pycon
    pip3.7 install spacy==2.0.18
    ```

    2.安装分词包，使用以下命令或登录对应网址下载分词包。 
    ```pycon
    wget https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz
    wget https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-2.0.0/de_core_news_sm-2.0.0.tar.gz
    ```
    解压并在分词包目录下，使用以下命令安装：
    ```pycon
    python3.7 setup.py install
    ```
3. 安装torchtext。
    ```pycon
    pip install torchtext==0.6
    ```


## 准备数据集

1. 获取数据集。

   初次执行脚本 `bash ./test/train_performance_1p.sh --data_path=数据集路径` 时，会将torchtext中multi30k数据集下载到`数据集路径`所对应文件目录下。

   以`--data_path=data`为例，初次运行后`data/multi30k`文件目录如下。

   ```
   ├── multi30k
       ├──test.de
       ├──test.en
       ├──train.de
       ├──train.en 
       ├──val.de  
       ├──val.en
   ```

<!--
2. 数据预处理（按需处理所需要的数据集）。

## 获取预训练模型（可选）

请参考原始仓库上的README.md进行预训练模型获取。将获取的bert\_base\_uncased预训练模型放至在源码包根目录下新建的“temp/“目录下。
-->

# <span id="xunlian">开始训练</span>

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。


   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/   
     ```
   - 单机单卡性能

     启动单卡性能。

     ```
     bash ./test/train_performance_1p.sh --data_path=/data/    
     ```

   - 单机8卡性能

     启动8卡性能。

     ```
     bash ./test/train_performance_8p.sh --data_path=/data/  
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_dir                          //数据集路径
   --addr                              //主机地址
   --workers                           //加载数据进程数      
   --epoch                             //重复训练次数
   --batch-size                        //训练批次大小
   --amp                               //是否使用混合精度
   --multiprocessing-distributed       //是否使用多卡训练
   --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# <span id="jieguo">训练结果展示</span>

**表 2**  训练结果展示表

| torch | Acc@1   | FPS | Npu_Nums | Epochs | AMP_Type |
|-----|---------|---------:|----------|----|---------:|
| 1.5 | -       | 1900 | 1        | 1  |       O2 |
| 1.5 | 12.8662 | 7397 | 8        | 10 |       O2 |
| 1.8 | -       | 1955.86 | 1        | 1  |       O2 |
| 1.8 | 12.5727 |  13693.3 | 8        | 10  |       O2 |


# 版本说明

## 变更

2022.11.30：更新torch1.8版本，重新发布。

2021.08.30：首次发布。

## 已知问题

无。