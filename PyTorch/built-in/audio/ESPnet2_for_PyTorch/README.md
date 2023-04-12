# ESPnet2 for PyTorch
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
  code_path=PyTorch/built-in/audio/
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

  在模型源码包根目录下执行命令，安装模型需要的依赖。

  ```
  pip3 install -r requirements.txt
  ```


- 安装ESPnet。

  1. 安装好相应的cann包、pytorch和apex包，并设置好pytorch运行的环境变量；

  2. 基于espnet官方的安装说明进行安装： [Installation — ESPnet 202205 documentation](https://espnet.github.io/espnet/installation.html) 

  安装过程比较复杂，需注意以下几点：

  - 安装依赖的软件包时，当前模型可以只安装cmake/sox/sndfile；

  - 跳过安装kaldi；

  - 安装espnet时，步骤1中的git clone ESPnet代码替换为下载本modelzoo中ESPnet的代码；步骤2跳过；步骤3中设置python环境，若当前已有可用的python环境，可以选择D选项执行；步骤4中进入tools目录后，需要增加installers文件夹的执行权限```chmod +x -R installers/```，然后直接使用make命令进行安装，不需要指定PyTorch版本；
  
  - custom tool installation这一步可以选择不安装。check installation步骤在make时已执行，可跳过；
  
  3. 运行模型前，还需安装：

  - boost: ubuntu上可使用 ```apt install libboost-all-dev```命令安装，centos上使用 ```yum install boost-devel``` 命令安装。
  - kenlm：进入<espnet-root>/tools目录，执行`make kenlm.done`
  
  4. 更新软连接：
  
      ```
      cd <espnet-root>/egs2/aishell/asr1
      rm -f asr.sh db.sh path.sh pyscripts scripts utils steps local/download_and_untar.sh
      ln -s ../../TEMPLATE/asr1/asr.sh asr.sh
      ln -s ../../TEMPLATE/asr1/db.sh db.sh
      ln -s ../../TEMPLATE/asr1/path.sh path.sh
      ln -s ../../TEMPLATE/asr1/pyscripts pyscripts
      ln -s ../../TEMPLATE/asr1/scripts scripts
      ln -s ../../../tools/kaldi/egs/wsj/s5/utils utils
      ln -s ../../../tools/kaldi/egs/wsj/s5/steps steps
      ln -s ../../../../egs/aishell/asr1/local/download_and_untar.sh local/download_and_untar.sh
      ```
      
  5. 增加执行权限：
  
     ```
     chmod +x -R ../../TEMPLATE/asr1
     chmod +x ../../../egs/aishell/asr1/local/download_and_untar.sh
     chmod +x -R local
     chmod +x run.sh
     ```


## 准备数据集

1. 获取数据集。

   本次训练采用**aishell-1**数据集，该数据集包含由 400 位说话人录制的超过 170 小时的语音，数据集目录结构参考如下所示。

   ```
   /downloads
          ├── data_aishell
          ├── data_aishell.tgz
          ├── resource_aishell
          └── resource_aishell.tgz
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。
   
   启动训练脚本stage 1 时自行下载并解压数据，下载时间较长，请耐心等待。 如果本地已有aishell数据集，可通过如下软连接命令进行指定。
   
   ```ln -s ${本地aishell数据集文件夹}/ downloads```


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练

     ```
     bash ./test/train_full_1p.sh --stage=起始stage  # 单卡精度
    
     bash ./test/train_performance_1p.sh --stage=起始stage  # 单卡性能
     ```
   
   - 单机8卡训练

     启动8卡训练
   
     ```
     bash ./test/train_full_8p.sh --stage=起始stage  # 8卡精度
    
     bash ./test/train_performance_8p.sh --stage=起始stage  # 8卡性能
     ```
   
3. 启动训练后，日志输出路径为：<espnet-root>/egs2/aishell/asr1/nohup.out ，该日志中会打印二级日志（各个stage日志）的相对路径。
如：stage 11 的日志路径为：“exp/asr_train_asr_conformer_raw_zh_char_sp/train.log”

模型训练脚本参数说明如下。

```shell
--stage   # 可选参数，默认为1，可选范围为：1~16。后续stage依赖前序stage，首次训练需从stage1开始。 
# stage 1 ~ stage 5 数据集下载与准备
# stage 6 ~ stage 9 语言模型训练
# stage 10 ~ stage 11 ASR模型训练
# stage 12 ~ stage 13 在线推理及精度统计
# stage 14 ~ stage 16 模型打包及上传
```


# 训练结果展示

**表 2**  训练结果展示表

| NAME          | CER_ERROR   | FPS    | Epochs | Torch_version |
|--------       |-------------|:-------|--------| :------------ |
| 1p-竞品       | -           | 196.86 | 1      | -             |
| 8p-竞品       | 95.4        | 398.8  | 50     | -             |
| 1p-NPU(非ARM)   | -           | 101.04 | 1      | 1.8           |
| 8p-NPU(非ARM)   | 95.5        | 399.7  | 50     | 1.8           |


# 版本说明

## 变更

2023.03.13：更新readme，重新发布。

2022.08.17：首次发布。

## FAQ

无。