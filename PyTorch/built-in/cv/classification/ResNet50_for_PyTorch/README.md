# ResNet50 for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

ResNet是ImageNet竞赛中分类问题效果较好的网络，它引入了残差学习的概念，通过增加直连通道来保护信息的完整性，解决信息丢失、梯度消失、梯度爆炸等问题，让很深的网络也得以训练。ResNet有不同的网络层数，常用的有18-layer、34-layer、50-layer、101-layer、152-layer。ResNet50的含义是指网络中有50-layer，由于兼顾了速度与精度，目前较为常用。

- 参考实现：

  ```
  url=https://github.com/pytorch/examples.git
  commit_id=49e1a8847c8c4d8d3c576479cb2fe2fd2ac583de
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
  | PyTorch 1.5 | pillow==8.4.0 |
  | PyTorch 1.8 | pillow==9.1.0 |
  
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

- 源码安装DLLogger库。
  ```
  下载源码链接： git clone https://github.com/NVIDIA/dllogger.git
  进入源码一级目录执行： python3 setup.py install
  ```

## 准备数据集


1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，将数据集上传到服务器任意路径下并解压。

   以ImageNet数据集为例，数据集目录结构参考如下所示。

   ```
   ├── ImageNet2012
         ├──train
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...   
              ├──...                     
         ├──val  
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                 │──图片1
                    │──图片2
                    │   ...                
   ```

   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。

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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

   - 单机单卡评测

     启动单卡评测。

     ```
     bash ./test/train_eval_1p.sh --data_path=/data/xxx/ --checkpoint=/checkpoint/xxx
     ```

   - 多机多卡性能数据获取流程。

     ```
     bash ./test/train_performance_multinodes.sh --data_path=数据集路径 --batch_size=单卡batch_size*所有卡数 --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
     如若遇到逻辑卡号与物理卡号不一致的情况，请手动在./test/train_performance_multinodes.sh中添加传参，例如--device-list='0,1,2,3'
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录,--checkpoint参数填写模型权重

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --amp                               //是否使用混合精度
   --data                              //数据集路径
   --addr                              //主机地址
   --seed                              //训练的随机数种子   
   --workers                           //加载数据进程数
   --learning-rate                     //初始学习率
   --weight-decay                      //权重衰减
   --print-freq                        //打印周期
   --dist-backend                      //通信后端
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小
   --benchmark                         //设置benchmark状态
   --dist-url                          //设置分布式训练网址
   --multiprocessing-distributed       //是否使用多卡训练
   --world-size                        //分布式训练节点数量
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME | Acc@1 | FPS       | Epochs   | AMP_Type | Torch_Version |
| :------: | :------:  | :------: | :------: | :------: | :------: |
| 1p-竞品A |   -    |   2065   |   1    | O2    | 1.5   |
| 8p-竞品A |   -    |  14268   |   90   | O2   | 1.5  |
| 1p-NPU |   -    | 1259.591 |   1    | O2 | 1.8 |
| 8p-NPU | 76.702 | 11898.83 | 90 | O2 | 1.8 |


# Resnet50-推理指导

-   [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)

  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集2)
  - [模型推理](#模型推理)

-   [模型推理性能&精度](#模型推理性能&精度)
-   [版本说明](#版本说明)

  ******

## 输入输出数据

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 256 x 256 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 1 x 1000 | FLOAT32  | ND           |

## 推理环境准备

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | >1.5.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手



1. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```


## 准备数据集2</a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型使用ImageNet 50000张图片的验证集，请前往ImageNet官网下载数据集

    ```
    ├── ImageNet
    |   ├── val
    |   |    ├── ILSVRC2012_val_00000001.JPEG
    │   |    ├── ILSVRC2012_val_00000002.JPEG
    │   |    ├── ......
    |   ├── val_label.txt
    ```

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   执行imagenet_torch_preprocess.py脚本，完成预处理。

   ```
   python3 imagenet_torch_preprocess.py resnet ./ImageNet/val ./prep_dataset
   ```
   每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成prep_dataset二进制文件夹
## 模型推理</a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       通过在线推理生成权重文件checkpoint.pth
   
   2. 导出onnx文件。
   
      1. 使用pth2onnx.py导出onnx文件。
   
         运行pth2onnx.py脚本。
   
         ```
         python3 pth2onnx.py ./checkpoint.pth
         ```
   
         获得resnet50_official.onnx文件。
   
   3. 使用ATC工具将ONNX模型转OM模型。
   
      1. 配置环境变量。
   
         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```
   
         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。
   
         ```
         npu-smi info
         #该设备芯片名为Ascend910A （请根据实际芯片填入）
         回显如下：
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       910A     | OK              | 69.5         40                0    / 0              |
         | 0       0         | 0000:82:00.0    | 0            950 / 15137                            |
         +===================+=================+======================================================+
         | 1       910A     | OK              | 65.3         36                0    / 0              |
         | 0       1         | 0000:89:00.0    | 0            1613 / 15137                            |
         +===================+=================+======================================================+
         ```
   
      3. 执行ATC命令。
   
         ```
         atc --model=resnet50_official.onnx --framework=5 --output=resnet50_bs64 --input_format=NCHW --input_shape="actual_input_1:64,3,224,224" --enable_small_channel=1 --log=error --soc_version=Ascend${chip_name} --insert_op_conf=aipp_resnet50.aippconfig
         
         备注：Ascend${chip_name}请根据实际查询结果填写
         ```
   
         - 参数说明：
         
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --insert\_op\_conf:  AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。
         
           运行成功后生成resnet50_bs64.om模型文件。
           
           

2.开始推理验证。

a.  安装ais_bench推理工具。

    请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

b.  执行推理。
   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh 
   python3 -m ais_bench --model ./resnet50_bs64.om --input ./prep_dataset/ --output ./ --output_dirname result --outfmt TXT
   ```

   - 参数说明：   
      - --model：模型地址
      -  --input：预处理完的数据集文件夹
      -  --output：推理结果保存地址
      -  --output_dirname: 推理结果保存文件夹
      -  --outfmt：推理结果保存格式
        
   运行成功后会在result/xxxx_xx_xx-xx-xx-xx（时间戳）下生成推理输出的txt文件。


c.  精度验证。

统计推理输出的Top 1-5 Accuracy
调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。
   ```
   python3 vision_metric_ImageNet.py ./result ./val_label.txt ./ result.json
   ```
   - 参数说明
     - val_label.txt：为标签数据
     - result.json：为生成结果文件

# 模型推理精度

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 
| --------- | ---------------- | ---------- | ---------- | 
| 910A | 64 | ImageNet | top-1: 76.96% <br>top-5: 93.24% |



# 版本说明

## 变更

2023.02.16：更新readme，重新发布。

2021.07.08：首次发布。

## FAQ

本模型单卡和多卡使用不同的脚本，脚本配置有差异， 会影响到线性度， 目前正在重构中。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md