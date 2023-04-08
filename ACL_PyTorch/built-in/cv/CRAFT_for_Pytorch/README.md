# CRAFT 模型-推理指导


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

CRAFT模型是一个文本检测模型。


- 参考实现：

  ```
  url=https://github.com/clovaai/CRAFT-pytorch.git
  commit_id=e332dd8b718e291f51b66ff8f9ef2c98ee4474c8
  model_name=CRAFT_for_Pytorch
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 640 x 640 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小                       | 数据排布格式 |
  | -------- | -------- | -------------------------- | ------------ |
  | y        | FLOAT32  | batchsize x 320 x 320 x 2  | ND           |
  | feature  | FLOAT32  | batchsize x 32 x 320 x 320 | ND           |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/clovaai/CRAFT-pytorch.git
   cd CRAFT-pytorch/
   git reset e332dd8b718e291f51b66ff8f9ef2c98ee4474c8 --hard
   cd ..
   ```
   
2. 安装依赖

   ```
   pip3 install -r requirements.txt
   ```
   

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型使用随机数据进行测试

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       到https://github.com/clovaai/CRAFT-pytorch 链接下下载General项的pretrained model craft_mlt_25k.pth
       
   2. 导出onnx文件。

      ```
      cp export_onnx.py CRAFT-pytorch/
      cp craft_mlt_25k.pth CRAFT-pytorch/
      cd CRAFT-pytorch
      python3 export_onnx.py --trained_model craft_mlt_25k.pth
      ```

      


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
         cd ..
         mkdir CRAFT-pytorch/output
         cp craft_atc.sh CRAFT-pytorch/
         cd CRAFT-pytorch
         bash craft_atc.sh --model {onnx_model} --bs {batch_size} --soc Ascend${chip_name}
         示例:
         bash craft_atc.sh --model craft --bs 1 --soc Ascend310P3
         ```
           - 参数说明：
             - --model：onnx模型名称
             - --bs：模型batchsize
             - --soc: 使用的卡

            运行成功后生成craft_{bs}.om模型文件。

4. 开始推理验证

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 精度验证。

      调用以下脚本，会打印出余弦相似度
      ```
      cd ..
      cp cosine_similarity.py CRAFT-pytorch/
      cd CRAFT-pytorch
      python3 cosine_similarity.py --bs {batch_size} --model_path {om_model_path}
      示例:
      python3 cosine_similarity.py --bs 1 --model_path ./output/craft_bs1.om
      ```
      - 参数说明：
        - --model_path：om模型所在路径
        - --bs：模型batchsize

   4. 性能验证
      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      cd output
      python3 -m ais_bench --model={om_model_path} --loop=1000 --batchsize={batch_size}
      示例:
      python3 -m ais_bench --model=craft_bs1.om --loop=1000 --batchsize=1
      ```

      - 参数说明：
        - --model：om模型
        - --batchsize：模型batchsize
        - --loop: 循环次数



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度 | 性能     |
| -------- |------------| ------ | ---- |--------|
|     310P3     | 1          |  随机数据  |   余弦相似度:0.999   | 132fps |
|     310P3     | 4          |  随机数据  |   余弦相似度:0.999   | 104fps |
|     310P3     | 8          |  随机数据  |   余弦相似度:0.999   | 103fps |
|     310P3     | 16         |  随机数据  |   余弦相似度:0.999   | 102fps  |
|     310P3     | 32         |  随机数据  |   余弦相似度:0.999   | 100fps  |
|     310P3     | 64         |  随机数据  |   余弦相似度:0.999   | 87fps  |

说明：模型的两个输出的余弦相似度与ONNX相比都是0.999