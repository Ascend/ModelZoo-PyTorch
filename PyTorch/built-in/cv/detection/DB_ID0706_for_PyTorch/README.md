# DBNet for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)



# 概述

## 简述

DB(Differentiable Binarization)是一种使用可微分二值图来实时文字检测的方法，
和之前方法的不同主要是不再使用硬阈值去得到二值图，而是用软阈值得到一个近似二值图，
并且这个软阈值采用sigmod函数，使阈值图和近似二值图都变得可学习。

- 参考实现：

  ```
  url=https://github.com/MhLiao/DB
  commit_id=4ac194d0357fd102ac871e37986cb8027ecf094e
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/detection
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

- 安装geos，可按照环境选择以下方式：

  1. ubuntu系统：

     ```
     sudo apt-get install libgeos-dev
     ```

  2. euler系统：

     ```
     sudo yum install geos-devel
     ```

  3. 源码安装：

     ```
     wget http://download.osgeo.org/geos/geos-3.8.1.tar.bz2
     bunzip2 geos-3.8.1.tar.bz2
     tar xvf geos-3.8.1.tar
     cd geos-3.8.1
     ./configure && make && make install
     ```

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
    
    请用户自行下载 `icdar2015` 数据集，解压放在任意文件夹 `datasets`下，数据集目录结构参考如下所示。

    ```
    |--datasets
       |--icdar2015
    ```

    > **说明：** 
    >该数据集的训练过程脚本只作为一种参考示例。

## 获取预训练模型

请用户自行获取预训练模型，将获取的 `MLT-Pretrain-Resnet50` 预训练模型，放至在源码包根目录下新建的 `path-to-model-directory` 目录下。


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
      1.安装环境，确认预训练模型放置路径，若该路径路径与model_path默认值相同，可不传参，否则执行训练脚本时必须传入model_path参数；
      2.开始训练
        bash ./test/train_full_1p.sh --data_path=${datasets} --model_path=${pretrain_model}    # 单卡精度
        bash ./test/train_performance_1p.sh --data_path=${datasets} --model_path=${pretrain_model}    # 单卡性能   
      ```
      **注意**：如果发现打屏日志有报checkpoint not found的warning，请再次检查预训练模型MLT-Pretrain-Resnet50的配置，以免影响精度。

   - 单机8卡训练

     启动8卡训练。

      ```
      1.安装环境，确认预训练模型放置路径，若该路径路径与model_path默认值相同，可不传参，否则执行训练脚本时必须传入model_path参数；
      2.开始训练
        bash ./test/train_full_8p.sh --data_path=${datasets} --model_path=${pretrain_model}    # 8卡精度
        bash ./test/train_performance_8p.sh --data_path=${datasets} --model_path=${pretrain_model}    # 8卡性能    
      ```
    
   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                          //数据集路径
   --addr                              //主机地址
   --num_workers                       //加载数据进程数      
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小，默认：240
   --lr                                //初始学习率
   --amp                               //是否使用混合精度
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME | Precision |  FPS | Epochs  |  AMP_Type | Torch_Version |
|:----:|:---------:|:----:|:------: | :-------: |:---:|
| 1P-竞品V | -       | - |        1 |       - | 1.5 |
| 8P-竞品V | -       | - |     1200 |       - | 1.5 |
| 1P-NPU-ARM | -         | 20.19 |        1 |       O2 | 1.8 |
| 8P-NPU-ARM | 0.907     |   88.073 |  1200 |       O2 | 1.8 |
| 1P-NPU-非ARM | -         | 20.265 |        1 |       O2 | 1.8 |
| 8P-NPU-非ARM | -    |   113.988 |  1200 |       O2 | 1.8 |


# 版本说明

## 变更

2022.12.23：Readme 整改。

## FAQ

无。


# DB模型PyTorch离线推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******

  

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

在基于分割的文本检测网络中，最终的二值化map都是使用的固定阈值来获取，并且阈值不同对性能影响较大。而在DB中会对每一个像素点进行自适应二值化，二值化阈值由网络学习得到，彻底将二值化这一步骤加入到网络里一起训练，这样最终的输出图对于阈值就会非常鲁棒。 


- 参考实现：

  ```
  url=https://github.com/MhLiao/DB 
  commit_id=4ac194d0357fd102ac871e37986cb8027ecf094e
  model_name=DB_for_PyTorch
  ```
  

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                       | 数据排布格式 |
  | -------- | -------- | -------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 736 x 1280 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小                       | 数据排布格式 |
  | -------- | -------- | -------------------------- | ------------ |
  | output1  | FLOAT32  | batchsize x 1 x 736 x 1280 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.6.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码

   ```shell
   git clone https://github.com/MhLiao/DB 
   cd DB
   git reset 4ac194d0357fd102ac871e37986cb8027ecf094e --hard
   patch -p1 < ../db.diff
   cd ..
   cp -r db_preprocess.py DB
   cp -r db_pth2onnx.py DB
   cp -r db_postprocess.py DB
   cd DB

   ```

2. 安装依赖

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集

   本模型支持icdar2015验证集。用户需自行获取数据集解压并上传数据集到DB/datasets路径下。目录结构如下：

   ```
   datasets/icdar2015/  
   ├── test_gts  
   ├── test_images  
   ├── test_list.txt  
   ├── train_gts  
   └── train_list.txt  
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行db_preprocess.py脚本，完成预处理

   ```shell
   python3 ./db_preprocess.py --image_src_path=./datasets/icdar2015/test_images --bin_file_path=./prep_dataset
   ```
   
   结果存在 ./prep_dataset 中


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      获取DB在线推理完成的权重文件MLT-Pretrain-Resnet50

   2. 导出onnx文件。

      1. 使用db_pth2onnx.py导出onnx文件

         ```shell
         python3 ./db_pth2onnx.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --resume ./MLT-Pretrain-Resnet50
         ```
         
         获得dbnet.onnx文件 
      
   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。
   
         ```sh
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```
   
      2. 执行命令查看芯片名称（$\{chip\_name\}）。
   
         ```sh
         npu-smi info
         #该设备芯片名为Ascend910A （自行替换）
         回显如下：
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       910A     | OK              | 15.8         42                0    / 0              |
         | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
         +===================+=================+======================================================+
         | 1       910A     | OK              | 15.4         43                0    / 0              |
         | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
         +===================+=================+======================================================+
         ```
   
      3. 执行ATC命令。
   
         ```sh
         atc --framework=5 --model=./dbnet.onnx --input_format=NCHW --input_shape="actual_input_1:${bs},3,736,1280" --output=db_bs${bs} --log=error --soc_version=Ascend${chip_name}
         ```
      
         运行成功后生成<u>***db_bs${bs}.om***</u>模型文件。
         
         - 参数说明
           
              - --model：为ONNX模型文件。
              
              - --framework：5代表ONNX模型。
              
              - --output：输出的OM模型。
              
              - --input\_format：输入数据的格式。
              
              - --input\_shape：输入数据的shape。
              
              - --log：日志级别。--soc\_version：处理器型号。
              
                
   
2. 开始推理验证

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```shell
        python3 -m ais_bench --model ./db_bs1.om --input ./prep_dataset  --output ./ --output_dirname result --device 0
        ```

        -   参数说明：

             -   --model：模型
             -   --input：数据位置
             -   --output：结果存的路径
             -   --output_dirname: 结果存的文件夹

        推理后的输出默认在当前目录result下。


   3. 精度验证。

      结果保存在result_bs1.json

      ```shell
      python3 ./db_postprocess.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --bin_data_path ./result --box_thresh 0.6 > result_bs1.json
      ```

      - 参数说明：

        - ./result：为生成推理结果所在路
        - result_bs1.json：为精度生成结果文件

   4. 性能验证

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```shell
        python3 -m ais_bench --model=db_bs${bs}.om --loop=20 --batchsize=${bs}
        ```

      - 参数说明：
        - --model：om模型

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size |  数据集   | 精度 |
| :------: | :--------: | :-------: | :--: | 
|  910A   |     1      | icdar2015 | 0.869 | 
