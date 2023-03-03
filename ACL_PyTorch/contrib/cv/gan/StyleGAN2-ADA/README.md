# StyleGAN2-ADA模型-推理指导


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

StyleGAN2-ADA是具有自适应鉴别器增强（ADA）的StyleGAN2，用有限的数据训练生成对抗网络，[论文链接](https://arxiv.org/abs/2006.06676)。

- 参考实现：

  ```
  url=https://github.com/NVlabs/stylegan2-ada-pytorch
  commit_id=765125e7f01a4c265e25d0a619c37a89ac3f5107
  code_path=
  model_name=StyleGAN2-ADA
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 512 | ND        |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 3 x 512 x 512| NCHW         |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git model
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   StyleGAN2-ADA网络使用随机生成隐变量作为输入来生成输出，生成方式见下一步。


2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行pkl2onnx.py脚本，完成预处理。

   ```
   python3 stylegan2-ada-pytorch_preprocess.py --num_input=200 --save_path=./pre_data
   ```
   生成`num_input个`随机输入，并保存为bin文件，保存目录为`./pre_data`。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pkl转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       权重文件为：[G_ema_bs8_8p_kimg1000.pkl](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/StyleGAN/PTH/G_ema_bs8_8p_kimg1000.pkl) 
       将获取的权重文件放在当前工作路径下。

   2. 导出onnx文件。

      1. 使用pkl2onnx.py导出onnx文件。

         运行pkl2onnx.py脚本。

         ```
         python3 pkl2onnx.py --batch_size=1 --pkl_file=./G_ema_bs8_8p_kimg1000.pkl --onnx_file=./G_ema_onnx_bs1.onnx
         python3 pkl2onnx.py --batch_size=16 --pkl_file=./G_ema_bs8_8p_kimg1000.pkl --onnx_file=./G_ema_onnx_bs16.onnx
         ```

         获得G_ema_onnx_bs1.onnx、G_ema_onnx_bs16.onnx文件。

  

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
            --model=./G_ema_onnx_bs${batchsize}.onnx \
            --output=G_ema_om_bs${batchsize} \
            --input_format=ND \
            --input_shape="z:${batchsize},512" \
            --log=error \
            --soc_version=Ascend${chip_name} \
            --buffer_optimize=off_optimize
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --buffer_optimize: 模型优化选项。

           运行成功后生成`G_ema_om_bs${batchsize}.om`模型文件。${batchsize}支持：1，16。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        python3 -m ais_bench --model=./G_ema_om_bs${batchsize}.om --input=./pre_data/ --output=./ --batchsize=${batchsize}
        ```

        -   参数说明：

             -   --model：om模型。
             -   --input：输入数据路径。
             -   --output：推理结果路径。
             -   --batchsize：om模型的batchsize。

        推理后的输出默认在当前目录下。


   3. 精度验证。

      a.调用`stylegan2-ada-pytorch_postprocess.py`脚本将om模型的推理结果转化为图像。

      ```
       python3 stylegan2-ada-pytorch_postprocess.py --bin_path=${output_path} --image_path=./results/om_bs${batchsize}
      ```

      - 参数说明：

             -   --bin_path：${output_path}为推理工具生成的推理结果路径。
             -   --image_path：为转化图像的生成路径，${batchsize}表示om模型的batchsize。

      运行后在`--image_path`指定的目录保存转化的图像。

      b.调用`perf_gpu.py`脚本使用pkl权重文件成生图像。

      ```
       python3 perf_gpu.py
      ```
      
      c.精度比对方法：将om模型的推理结果转化的图像与pkl权重文件成生图像进行对比，两幅图像在视觉效果上一致。


   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3.7 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型
        - --batchsize：om模型的batchsize。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|  Ascend310 | 1                 | 随机生成数据           | 图片视觉评价           | 19.27 fps                |
|  Ascend310P3 | 1                 | 随机生成数据           | 图片视觉评价           | 39.81 fps                |
|  Ascend310P3 | 16                 | 随机生成数据           | 图片视觉评价           | 37.06 fps                |