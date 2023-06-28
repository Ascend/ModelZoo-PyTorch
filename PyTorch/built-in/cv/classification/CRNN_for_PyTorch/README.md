# CRNN for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

CRNN (Convolutional Recurrent Neural Network) 于2015年由华中科技大学的白翔老师团队提出，直至今日，仍旧是文本识别领域最常用也最有效的方法。

- 参考实现：

  ```
  url=https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec.git
  commit_id=90c83db3f06d364c4abd115825868641b95f6181
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.2.2.post3；pillow==8.4.0 |
  | PyTorch 1.8 | torchvision==0.9.1；pillow==9.1.0 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。

## 准备数据集

1. 获取数据集。

   模型训练以 MJSynth 数据集为训练集，IIIT 数据集为测试集。

   用户自行下载并解压 data_lmdb_release.zip，将其中的data_lmdb_release/training/MJ/MJ_train 文件夹 (重命名为 MJ_LMDB) 和 
   data_lmdb_release/evaluation/IIIT5k_3000 文件夹 (重命名为 IIIT5k_lmdb)上传至服务器的任意目录下，作为数据集目录。
   > 注意：若用户选择下载原始数据集，则需要将其转换为 lmdb 格式数据集，再根据上述步骤进行数据集上传。

   数据集目录结构参考如下所示：
   ```
   ├──服务器任意目录下
       ├──MJ_LMDB
             │──data.mdb
             │──lock.mdb
       ├──IIIT5K_lmdb
             │──data.mdb
             │──lock.mdb
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
     bash ./test/train_performance_1p.sh --data_path=数据集路径    # 单卡性能
     
     bash ./test/train_full_1p.sh --data_path=数据集路径           # 单卡精度 
     ```

   - 单机8卡训练

     启动8卡训练。
     ```
     bash ./test/train_performance_8p.sh --data_path=数据集路径    # 8卡性能
     
     bash ./test/train_full_8p.sh --data_path=数据集路径           # 8卡精度
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                         //训练集路径
   --epochs                            //重复训练次数
   --npu                               //npu训练卡id设置
   --max_step                          //设置最大迭代次数
   --stop_step                         //设置停止的迭代次数
   --profiling                         //设置profiling的方式
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|  NAME  | Acc@1 |    FPS    | Epochs | AMP_Type | Torch_Version |
| :----: | :---: | :-------: | :----: | :------: | :-----------: |
| 1p-NPU |   -   | 11733.53  |   1    |    O2    |      1.8      |
| 8p-NPU | 0.75  | 106510.27 |  100   |    O2    |      1.8      |


# 版本说明
2022.02.17：更新readme，重新发布。

## FAQ
无。

# CRNN-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)
  
- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

  ******




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

文字识别是图像领域一个常见问题。对于自然场景图像，首先要定位图像中的文字位置，然后才能进行文字的识别。对定位好的文字区域进行识别，主要解决的问题是每个文字是什么，将图像中的文字区域进转化为字符信息。CRNN全称为Convolutional Recurrent Neural Network，是一种卷积循环神经网络结构，用于解决基于图像的序列识别问题，特别是场景文字识别问题。主要用于端到端地对不定长的文本序列进行识别，不用先对单个文字进行切割，而是将文本识别转化为时序依赖的序列学习问题，也就是基于图像的序列识别。


- 参考实现：

  ```
  https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec
  branch=master
  commit_id=90c83db3f06d364c4abd115825868641b95f6181
  ```

  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                     | 数据排布格式 |
  | -------- | -------- | ------------------------ | ------------ |
  | input    | FLOAT32  | batchsize x 1 x 32 x 100 | NCHW         |


- 输出数据

  | 输出数据 | 大小                | 数据类型 | 数据排布格式 |
  | -------- | ------------------- | -------- | ------------ |
  | output1  | 26 x batchsize x 37 | FLOAT32  | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.12.1  | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>



1. 安装依赖。

   ```
   pip3 install -r requirments.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持多种开源OCR mdb数据集（例如IIIT5K_lmdb），请用户自行准备好图片数据集，IIIT5K_lmdb验证集目录参考。

    ```
   ├── IIIT5K_lmdb        # 验证数据集
     ├── data.mdb         # 数据文件
     └── lock.mdb         # 锁文件
    ```

2. 数据预处理。

   1. 执行parse_testdata.py脚本。
   
   ```
   python3 parse_testdata.py ./IIIT5K_lmdb input_bin
   ```
   
   执行成功后，二进制文件生成在*./input_bin*文件夹下，标签数据label.txt生成在当前目录下。


## 模型推理<a name="section741711594517"></a>

模型转换。

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

获取权重文件。

导出onnx文件。

1. 使用pth2onnx.py导出onnx文件。

   运行pth2onnx.py脚本。

   ```
   python3.7 pth2onnx.py ./checkpoint.pth ./crnn_npu_dy.onnx
   ```

   获得crnn_npu_dy.onnx文件。

使用ATC工具将ONNX模型转OM模型。

2. 配置环境变量。

   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

   > **说明：** 
   > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

3. 执行命令查看芯片名称（$\{chip\_name\}）

      ```shell
      npu-smi info
      ```
      该设备芯片名为Ascend910A （请根据实际芯片填入）

4. 执行atc命令

      ```shell
      # Ascend${chip_name}请根据实际查询结果填写 
      atc --model=crnn_npu_dy.onnx --framework=5 --output=crnn_final_bs16 --input_format=NCHW --input_shape="actual_input_1:16,1,32,100" --log=error --soc_version=Ascend${chip_name}
      ```
      
      参数说明:  
      
      - --model：为ONNX模型文件 
      
      - --framework：5代表ONNX模型 
      
      - --output：输出的OM模型 
      
      - --input_format：输入数据的格式 
      
      - --input_shape：输入数据的shape 
      
      - --log：日志级别 
      
      - --soc_version：处理器型号 
      
      运行成功后生成crnn_final_bs16.om模型文件 

5. 开始推理验证。

    a. 安装ais_bench推理工具。

    请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

    b. 执行推理

    ```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    python3 -m ais_bench --model ./crnn_final_bs16.om --input ./input_bin --output ./ --output_dirname result --device 0 --batchsize 16 --output_batchsize_axis 1
    ```

    参数说明:

    - --model：模型地址 
    - --input：预处理完的数据集文件夹 
    - --output：推理结果保存路径
    - --output_dirname: 推理结果存储位置

    运行成功后会在 ./result 下生成推理输出的bin文件


    c. 精度验证。
    运行脚本postpossess_CRNN_pytorch.py进行精度测试，精度会打屏显示。

    ```
    python3 postpossess_CRNN_pytorch.py ./result ./label.txt
    ```
6. 性能验证

    可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
    ```
    python3.7 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
    ```
    - 参数说明
      - --model：om模型
      - --loop：循环次数
      - --batchsize：推理张数

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集      | 精度   |
| -------- | ---------- | ----------- | ------ |
| 910A    | 16         | IIIT5K_lmdb | 76.57% |


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
