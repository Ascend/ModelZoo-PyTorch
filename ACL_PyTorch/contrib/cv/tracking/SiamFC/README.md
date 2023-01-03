### SiamFC模型-推理指导


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

SiamFC是视觉目标跟踪领域首次采用孪生网络方法的模型，该模型将经典的特征提取网络AlexNet与孪生网络相结合，网络采用全卷积的方式对模板图片与搜索图片进行卷积计算，以在搜索图片上找出最符合模板图片的位置。 


- 参考论文：

  [Bertinetto， Luca， et al. “用于对象跟踪的全卷积连体网络”。欧洲计算机视觉会议。施普林格，湛，2016年。](https://link.springer.com/chapter/10.1007/978-3-319-48881-3_56) 
  
- 参考实现：

  ```
  url=https://github.com/HonglinChu/SiamTrackers/tree/master/2-SiamFC/SiamFC-VID
  commit_id=8471660b14f970578a43f077b28207d44a27e867
  code_path=/ACL_PyTorch/contrib/cv/tracking/SiamFC
  model_name=SiamFC
  ```
  



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input1   | RGB_FP32 | batchsize x 3 x 255x 255  | NCHW         |
  | input2   | RGB_FP32 | batchsize x 9 x 127 x 127 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小              | 数据排布格式 |
  | -------- | -------- | ----------------- | ------------ |
  | output1  | FLOAT32  | 3 x 256 x 6 x 6   | NCHW         |
  | output2  | FLOAT32  | 1 x 768 x 22 x 22 | NCHW         |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.9.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

  


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 安装依赖。

   ```
   pip3 install -r requirements.txt  
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持[OTB2015](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html) 约60000张图片的测试集。请用户需自行获取OTB2015数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser）。本模型将使用到OTB2015数据集上的100个测试序列，并计算Success_score和Precision_score。 目录结构如下：

   ```
   OTB2015
   ├── Basketball
   │   ├── groundtruth_rect.txt
   │   └── img
   │       ├── 0001.jpg
   │       ├── 0002.jpg
   │       ├── 0003.jpg
   │       ├── 0004.jpg
   │       ├── 0005.jpg
   │       ├── 0006.jpg
   ...........
   ├── Walking
   │   ├── groundtruth_rect.txt
   │   └── img
   │       ├── 0001.jpg
   │       ├── 0002.jpg
   │       ├── 0003.jpg
   │       ├── 0004.jpg
   │       ├── 0005.jpg
   │       ├── 0006.jpg
   │       ├── 0007.jpg
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   由于本模型的特殊性，将不能使用AIPP功能进行推理，同时也无法简单地进行批量离线推理，而需要对每一帧都单独地进行数据预处理，并在预处理后立即生成数据集info文件、推理、后处理。因此，下面仅简要介绍数据预处理步骤。

   该模型包含两个分支，分别对模板图片和搜索图片进行处理，详见prepostprocess.py中的cropexemplar和cropsearch。具体操作包括数据增强、图片裁剪、写入bin文件。该步骤没有具体执行命令，其过程包括在推理验证2.3的命令中。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       从源码包中获取权重文件[siamfc.pth](https://pan.baidu.com/s/1N3Igj4ZgntjRevsGA5xOTQ)，提取码： 4i4l ， 放置于本代码仓./pth目录下 。

   2. 导出onnx文件。
   
       1. 使用pth/siamfc.pth导出onnx文件。

          运行pth2onnx.py脚本。
   
          ```
          mkdir onnx
          python3 pth2onnx.py pth/siamfc.pth onnx/exemplar.onnx onnx/search.onnx
          ```
          
          获得exemplar.onnx和search.onnx文件，均在onnx目录下。
   
   3. 使用ATC工具将ONNX模型转OM模型。
   
         1. 配置环境变量。
   
            ```
            source /usr/local/Ascend/ascend-toolkit/set_env.sh
            ```
   
         2. 执行命令查看芯片名称（$\{chip\_name\}）。
   
            ```
            npu-smi info
            #该设备芯片名为Ascend310P3 （自行替换）
            回显如下：
            +-------------------+-----------------+------------------------------------------------------+
            | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
            | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
            +===================+=================+======================================================+
            | 0       310P3     | OK              | 15.8         42                0    / 0              |
            | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
            +===================+=================+======================================================+
            | 1       310P3     | OK              | 15.4         43                0    / 0              |
            | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
            +===================+=================+======================================================+
            ```
   
         3. 执行ATC命令。
   
            ```
            atc --model=./onnx/exemplar.onnx --framework=5 --output=./om/exemplar_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,127,127" --log=debug --soc_version=Ascend${chip_name}
            
            atc --model=./onnx/search.onnx --framework=5 --output=./om/search_bs1 --input_format=NCHW --input_shape="actual_input_1:1,9,255,255" --log=debug --soc_version=Ascend${chip_name}
            ```
   
            - 参数说明：
   
              -   --model：为ONNX模型文件。
              -   --framework：5代表ONNX模型。
              -   --output：输出的OM模型。
              -   --input\_format：输入数据的格式。
              -   --input\_shape：输入数据的shape。
              -   --log：日志级别。
              -   --soc\_version：处理器型号。
   
            运行成功后获得exemplar_bs1.om和search_bs1.om文件，均在om目录下。
   
2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        执行数据预处理脚本。
   
        ```
       python3 get_perf_data.py ./pre_dataset1 ./pre_dataset2
        ```

        执行ais_infer工具进行性能验证。
   
        ```
        python3.7 -m ais_bench  --model ./om/exemplar_bs1.om --input pre_dataset1/ --device 0 --batchsize 1
        python3.7 -m ais_bench  --model ./om/search_bs1.om --input pre_dataset2/ --device 0 --batchsize 1
        ```
   
        -   参数说明：
   
             -   --model：om模型。
             -   --input：预处理数据集路径。
             -   --output：推理结果所在路径。
             -   --outfmt：推理结果文件格式。
             -   --batchsize：不同的batchsize。

        
   3. 精度验证。
   
      调用wholeprocess.py脚本进行精度验证，由于此网络性能验证时需要两个模型交替运行，所以使用脚本进行推理。
   
      ```
      python3 wholeprocess.py ./OTB2015/ ./pre_dataset ./dataset_info ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py ./om/exemplar_bs1.om ./om/search_bs1.om ${batchsize}
      ```
   
      - 参数说明：
   
        - ./OTB2015/：数据地址。
        - ./pre_dataset：数据预处理的保存地址。
        - ./dataset_info： 数据信息的保存地址。
        - ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py：推理工具所在路径。
        - ./om/exemplar_bs1.om：exemplar模型所在路径。
        - ./om/search_bs1.om：search_bs1模型所在路径。
        - ${batchsize}：不同的batchsize。
   
      >**注：** 
      >该模型预处理、推理及验证精度均在wholeprocess.py脚本中进行，验证性能使用get_perf_data.py脚本生成假数据进行验证。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

精度对比：

| Model     | SiamFc                                        |
| --------- | --------------------------------------------- |
| 开源精度  | success_score: 0.576   precision_score: 0.767 |
| 310P3精度 | success_score: 0.572   precision_score: 0.762 |

性能对比：

| 芯片型号 | Batch Size   | 数据集 | 模型 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| 310P3 | 1 | OTB2015 | exemplar_bs1.om | 6072.648 |
| 310P3 | 1 | OTB2015 | search_bs1.om | 948.314 |