# CSPResNeXt50模型-推理指导


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

CSPResNeXt50提出了跨阶段局部网络（CSPNet），用来缓解以往工作需要从网络架构角度进行大量推理计算的问题，把这个问题归结为网络优化中的[重复梯度信息]。CSPNet通过整合网络阶段开始和结束的特征图来尊重梯度的可变性，在实验中，该方法减少了20%的计算量。



- 参考实现：

  ```
  url=https://github.com/rwightman/pytorch-image-models
  commit_id=d584e7f617a4d0f1a0b4838227bd1f8852dfa236
  model_name=CSPResNeXt50
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | ND         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 1000 | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动 

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |





# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/rwightman/pytorch-image-models.git
   cd pytorch-image-models            
   git reset --hard d584e7f617a4d0f1a0b4838227bd1f8852dfa236
   cd ..
   ```

2. 安装依赖。

   ```
   pip install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型使用[ImageNet官网](https://gitee.com/link?target=http%3A%2F%2Fwww.image-net.org)的5万张验证集进行测试，以ILSVRC2012为例，用户需获取[ILSVRC2012数据集](http://www.image-net.org/download-images)，并上传到服务器，图片与标签分别存放在./imagenet/val与./imageNet/val_label.txt。
   ```
   ├── imagenet
       ├── val
       ├── val_label.txt 
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行cspresnext_preprocess.py脚本，完成预处理。

   ```
   python cspresnext_preprocess.py ./imageNet/val ./prep_dataset 
   ```
   - 参数说明：

      -   第一个参数：数据集
      -   第二个参数：预处理保存地址

   

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. [获取权重文件](https://drive.google.com/open?id=1IxKu5lAYCo4UpzAl5pOBIDAvC843vjn1)
     cspresnext50_ra_224-648b4713.pth

   2. 导出onnx文件。

      1. 使用cspresnext_pth2onnx.py导出onnx文件。
         ```
         python cspresnext_pth2onnx.py --pth=cspresnext50_ra_224-648b4713.pth --onnx=cspresnext.onnx
         ```
         - pth：pth权重文件
         - onnx：onnx文件名
         
         获得cspresnext.onnx文件。



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
         atc --framework=5 \
             --model=./cspresnext.onnx \
             --input_format=NCHW \
             --input_shape="image:${bs},3,224,224" \
             --output=cspresnext_bs${bs} \
             --log=debug \
             --soc_version=Ascend${chip_name} \
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***cspresnext_bs${bs}.om***</u>模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```
      python -m ais_bench --model=cspresnext_bs${bs}.om --input=./prep_dataset/ --output=./ --output_dirname=result --outfmt=TXT --batchsize=${bs}
        ```

        -   参数说明：
             -   model：om文件路径。
             -   input：预处理文件。
             -   output：推理文件保存路径
             -   output_dirname：推理文件保存目录

        推理后的输出保存在当前目录result下。


   3. 精度验证。

      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据

      ```
      python3 cspresnext_postprocess.py --result_path=./result_summary.json --gtfile_path=./imageNet/val_label.txt
      ```

      - 参数说明：
        - result_path：推理生成结果信息的json文件
        - gtfile_path：数据集标签


   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python -m ais_bench --model=cspresnext_bs${bs}.om --loop=100 --batchsize=${bs}
        ```

      - 参数说明：
        - --model：om文件路径。
        - --batchsize：batchsize大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|   Ascend310P3        |         1         |      ImageNet      |    79.79%        |       1498          |
|   Ascend310P3        |         4         |      ImageNet      |    79.79%        |       3251          |
|   Ascend310P3        |         8         |      ImageNet      |            |       2998          |
|   Ascend310P3        |         16         |      ImageNet      |           |       2454          |
|   Ascend310P3        |        32        |      ImageNet      |           |       2289          |
|   Ascend310P3        |         64         |      ImageNet      |           |      1888           |