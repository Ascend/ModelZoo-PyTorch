# GMA模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)





# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

GMA着重解决光流估计中被遮挡点的光流估计问题。GMA定义的遮挡点是在当前帧中可见但在下一帧中不可见的点。以前的工作依赖CNN来学习遮挡，但收效不大，或者需要多帧并使用时间平滑度来推理遮挡。GMA通过对图像自相似性进行建模，来更好地解决遮挡问题。



- 参考实现：

  ```
  url=https://github.com/zacjiang/GMA
  commit_id=2f1fd29468a86a354d44dd25d107930b3f175043
  model_name=GMA
  ```





## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input1    | RGB_FP32 | batchsize x 3 x 440 x 1024 | NCHW         |
  | input2    | RGB_FP32 | batchsize x 3 x 440 x 1024 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 3 x 440 x 1024 | NCHW           |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.12.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/zacjiang/GMA.git
   cd GMA
   git reset --hard 2f1fd29468a86a354d44dd25d107930b3f175043
   cd ..
   patch -p2 < GMA.patch
   ```

2. 安装依赖。

   ```
   pip3 install -r requirement.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持MPI-Sintel-complete验证集。
   ```
   wget http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip --no-check-certificate
   ```
   将MPI_Sintel-complete文件夹解压并上传数据集到源码包路径data文件夹下。目录结构如下：

   ```
   data
   ├─ test 
   ├─ training
   ├─ flow_code
   └─ bundler
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行GMA_preprocess.py脚本，完成预处理。

   ```
   python3 GMA_preprocess.py --dataset ./GMA/data/training --output ./data_preprocessed_clean --status clean

   python3 GMA_preprocess.py --dataset ./GMA/data/training --output ./data_preprocessed_final --status final
   ```
   - 参数说明：

      -   dataset：数据集地址
      -   output：预处理数据保存地址
      -   status：预处理数据模式

   


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       权重文件在源码仓checkpoint文件夹中

   2. 导出onnx文件。

      1. 使用GMA_pth2onnx.py导出onnx文件。

         运行GMA_pth2onnx.py脚本。

         ```
         python3 GMA_pth2onnx.py --input_file=./GMA/checkpoints/gma-sintel.pth --output_file=GMA_${bs}.onnx --batchsize=${bs}
         ```
         - 参数说明：

            -   input_file：权重文件
            -   output_file：onnx模型名称
            -   batchsize：batchsize大小         

         获得GMA.onnx文件。

      2. 请访问[auto-optimizer改图工具](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)代码仓，根据readme文档进行工具安装。

         ```
         python3 -m auto_optimizer optimize GMA_${bs}.onnx GMA_m_${bs}.onnx -k 4
         ```

         获得GMA_m.onnx文件。


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
         atc --framework=5 --input_shape="image1:${bs},3,440,1024;image2:${bs},3,440,1024" --output=GMA_bs${bs} --soc_version=Ascend${chip_name} --keep_dtype=keep.cfg --model=GMA_m_${bs}.onnx 
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_shape：输入数据的shape。
           -   --soc\_version：处理器型号。
           -   --keep_dtype:指定部分算子fp32

           运行成功后生成<u>***GMA_bs${bs}.om***</u>模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```
        mkdir -p ./output_clean/bs${bs}
        python3 -m ais_bench --model=GMA_bs${bs}.om --batchsize=${bs} --input='data_preprocessed_clean/image1,data_preprocessed_clean/image2' --output=./output_clean  --output_dirname=bs${bs}

       mkdir ./output_final/bs${bs}
       python3 -m ais_bench --model=GMA_bs${bs}.om --batchsize=${bs} --input='data_preprocessed_final/image1,data_preprocessed_final/image2' --output=./output_final --output_dirname=bs${bs}  
        ```

        -   参数说明：

             -   model：om文件路径。
             -   input：输入预处理数据
             -   output：推理结果保存路径


   3. 精度验证。

      调用GMA_postprocess.py脚本计算精度

      ```
      python3 GMA_postprocess.py --gt_path=./data_preprocessed_clean/gt --output_path=./output_clean/bs${bs}
      python3 GMA_postprocess.py --gt_path=./data_preprocessed_final/gt --output_path=./output_final/bs${bs}
      ```

      - 参数说明：

        - gt_path:标杆数据文件夹

        - output_path:推理输出数据


        

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
       python3 -m ais_bench --model=GMA_bs${bs}.om --loop=100 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型
        - --batchsize：batchsize大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|    Ascend310P3       |         1         |     Sintel       |     final：88.95%(1px)    clean:92.65%(1px)       |        0.766         |
|    Ascend310P3       |        4         |     Sintel       |            |      0.772           |
|    Ascend310P3       |         8         |     Sintel       |            |        0.678         |
|    Ascend310P3       |         16         |     Sintel       |     内存超过，无法推理       |                 |