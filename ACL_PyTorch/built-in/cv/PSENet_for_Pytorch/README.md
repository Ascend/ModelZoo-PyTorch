# PSENet模型-推理指导


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

PSENet(渐进式的尺度扩张网络)是一种文本检测器，能够很好地检测自然场景中的任意形状的文本。该网络提出的方法能避免现有bounding box回归的方法产生的对弯曲文字的检测不准确的缺点，也能避免现有的通过分割方法产生的对于文字紧靠的情况分割效果不好的缺点。该网络是从FPN中受到启发采用了U形的网络框架，先通过将网络提取出的特征进行融合，然后利用分割的方式将提取出的特征进行像素分类，最后利用像素的分类结果通过一些后处理得到文本检测结果。


- 参考实现：

  ```
  model_name=built-in/cv/PSENet_for_Pytorch
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 704 x 1216 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 7 x 704 x 1216 | NCHW           |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.6.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>


1. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）



   本模型支持ICDAR2015数据集。用户需自行获取数据集，其目录结构如下：

   ```
   ICDAR2015
   ├── gt.zip    //验证集标注信息       
   └── val2015             // 验证集文件夹
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行`preprocess_psenet_pytorch.py`脚本，完成预处理。

   ```
   python3 preprocess_psenet_pytorch.py.py ./ICDAR2015/val2015 ./prep_bin
   ```

  


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       ```shell
       wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/PSENet/PTH/PSENet_for_PyTorch_1.2.pth
       ```

   2. 导出onnx文件。

      1. 使用`export_onnx.py`导出onnx文件。
         运行XXX脚本。

         ```
         python3 export_onnx.py PSENet_for_PyTorch_1.2.pth
         ```

         获得`PSENet_704_1216_nearest.onnx`文件。


   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         会显如下：
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

         ```shell
         atc \
            --model= PSENet_704_1216_nearest.onnx \
            --framework=5 \
            --output=PSENet_704_1216_nearest_bs1 \
            --input_format=NCHW \
            --input_shape="actual_input_1:1,3,704,1216" \
            --enable_small_channel=1 \
            --log=error \
            --soc_version=Ascend${chip_name}
          
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --enable_small_channel:使能小通道
           -   --soc\_version：处理器型号。
          

           运行成功后生成`PSENet_704_1216_nearest_bs1.om`模型文件。

2. 开始推理验证。
   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais-bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

   2. 执行推理。

        ```
        python3 -m ais_bench  \
               --model PSENet_704_1216_nearest_bs1.om \
               --input ./prep_bin \
               --output ./
               --output_dirname result
        ```

        -   参数说明：

             -   --model：om文件
             -   --input：输入文件
             -   --output：推理结果保存路径
             -   --output_dirname:推理结果保存文件夹
                  	
        推理后的输出默认在当前目录`result`下。

        >**说明：** 
        >执行ais-bench工具请选择与运行环境架构相同的命令。

   3. 精度验证。

      1. 调用`pth_bintotxt_nearest.py`脚本进行后处理，为计算精度做准备

         ```
         python3 pth_bintotxt_nearest.py ./ICDAR2015/val2015 ./result ./txt
         ```

         - 参数说明：
            - ./ICDAR2015/val2015：原始图片路径 
            - ./result：推理结果路径
            - ./txt：解析结果保存路径
      2. 精度计算

         ```shell
         cd ./txt
         zip -r ../npu_res.zip ./*
         cd ../Post-processing
         python3 script.py -g ../ICDAR2015/gt.zip -s ../npu_res.zip
         ```

         - 参数说明：
            - -g:label真值zip包
            - -s:NPU预测结果zip包
         
         结果会打屏显示


   4. 性能验证。

      可使用ais-bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型
        - --batchsize：模型batchsize
        - --loop: 循环次数



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度 | 性能 |
| -------- | ---------- | ------ | ---- | ---- |
|     310P3     |     1       |   ICDAR2015     |   acc:0.805<br>recall:0.639   |   70   |
|     310P3     |     4      |   ICDAR2015     |   acc:0.805<br>recall:0.639   |   64   |
|     310P3     |     8       |   ICDAR2015     |   acc:0.805<br>recall:0.639   |   63   |
|     310P3     |     16       |   ICDAR2015     |   acc:0.805<br>recall:0.639   |   62   |
|     310P3     |     32       |   ICDAR2015     |   acc:0.805<br>recall:0.639   |   60   |
|     310P3     |     64       |   ICDAR2015     |   acc:0.805<br>recall:0.639   |   54   |


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md