# ESPnet Dynamic for PyTorch
- [概述](#概述)
- [准备训练环境](#准备训练环境)
- [开始训练](#开始训练)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)

# 概述
ESPNet是一套基于E2E的开源工具包，可进行语音识别等任务。从另一个角度来说，ESPNet和HTK、Kaldi是一个性质的东西，都是开源的NLP工具；引用论文作者的话：ESPnet是基于一个基于Attention的编码器-解码器网络，另包含部分CTC组件。

- 参考实现：

  ```
  url=https://github.com/espnet/espnet/tree/v.0.10.5
  commit_id=b053cf10ce22901f9c24b681ee16c1aa2c79a8c2
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/audio
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

  ```
  pip3 install -r requirements.txt
  ```
- 安装ESPnet。

  1. 安装好相应的cann包、pytorch和apex包，并设置好pytorch运行的环境变量；

  2. 基于espnet官方的安装说明进行安装： [Installation — ESPnet 202205 documentation](https://espnet.github.io/espnet/installation.html) 

     安装过程比较复杂，需注意以下几点：

     - 安装依赖的软件包时，当前模型可以只安装cmake/sox/sndfile ；

     - 安装kaldi时，当前模型调测选择了OpenBLAS作为BLAS库，在compile kaldi & install阶段，使用如下命令安装：

       ```
       $ cd <kaldi-root>/src
       $ ./configure --openblas-root=../tools/OpenBLAS/install --use-cuda=no
       $ make -j clean depend; make -j <NUM-CPU>
       ```

     - 安装espnet时，步骤1中的git clone ESPnet代码替换为下载本modelzoo中ESPnet的代码；步骤3中设置python环境，若当前已有可用的python环境，可以选择D选项执行；步骤4中进入tools目录后，直接使用make命令进行安装，不需要指定PyTorch版本;

     - custom tool installation这一步可以选择不安装。最后通过check installation步骤检查安装结果；

  3. 运行模型前，还需安装：

  - boost: ubuntu上可使用 apt install libboost-all-dev命令安装，其它系统请选择合适命令安装
  - kenlm：进入<espnet-root>/tools目录，执行make kenlm.done
  
  4. (可选)更新软连接：

    ```
      cd <espnet-root>/egs/aishell/asr1
      rm -f utils steps
      ln -s ../../../tools/kaldi/egs/wsj/s5/utils utils
      ln -s ../../../tools/kaldi/egs/wsj/s5/steps steps
      cd <espnet-root>/egs/aishell/asr1/conf
      rm -f train.yaml decode.yaml
      ln -s tuning/train_pytorch_conformer_kernel15.yaml train.yaml
      ln -s tuning/decode_pytorch_transformer.yaml decode.yaml
     ```


## 准备数据集

1. 获取数据集。

   用户自行下载 `aishell-1` 数据集，并将下载好的数据集放置服务器的任意目录下。该数据集包含由 400 位说话人录制的超过 170 小时的语音。数据集目录结构参考如下所示。

   ```
    aishell-1
       ├── data_aishell.tgz
       |
       └── resource_aishell.tgz
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。


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
     bash ./test/train_full_1p.sh --stage=起始stage --data_path=/data/xxx/  # 单卡精度
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。
    
     ```
     bash ./test/train_full_8p.sh --stage=起始stage --data_path=/data/xxx/  # 8卡精度
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```
   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```shell
   --stage              //模型训练的起始阶段，默认为-1，即从数据下载开始启动训练。若之前数据下载、准备、特征生成等阶段已完成，可配置--stage=3开始训练。
   --stop_stage         //模型训练的终止阶段
   --data               //数据集路径
   --ngpu               //训练设备卡数量
   --test_output_dir    //输出路径
   ```
   > **说明：**
   >--stage <-1 ~ 5>、--stop_stage <-1 ~ 5>：控制模型训练的起始、终止阶段。模型包含 -1 ~ 5 训练阶段，其中 -1 ~ 2 为数据下载、准备、特征生成等阶段，3为LM训练，4为ASR训练，5为decoding。首次运行时请从 -1 开始，-1 ~ 2 阶段执行过一次之后，后续可以从stage 3 开始训练。LM和ASR是在NPU上运行的，其余都在CPU上运行。

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Error | FPS(iters/sec)  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V |  -   | LM：24.154  |   -    |    LM:O1 <br> ASR:O2     |      1.5      |
| 8p-竞品V | test数据集：6.1 <br> dev数据集：5.6 | ASR：1.1966 |  -  |    LM:O1 <br> ASR:O2     |      1.5      |
|  1p-NPU  |   -   | LM：7.3368  |   -    |    LM:O1 <br> ASR:O2    |      1.8      |
|  8p-NPU  |  test数据集：6.1 <br> dev数据集：5.6  | ASR：0.77794  |  -   |    LM:O1 <br> ASR:O2    |      1.8      |


# 版本说明

## 变更

2022.08.17：首次发布。

## FAQ

无。


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md







