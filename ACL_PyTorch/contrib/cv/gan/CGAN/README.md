# CGAN模型-推理指导


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

CGAN(条件生成对抗网络,Conditional Generative Adversarial Nets）是生成对抗网络(GAN)的条件版本.可以通过简单地向模型输入数据来构建.在无条件的生成模型中，对于生成的数据没有模式方面的控制，很有可能造成模式坍塌．而条件生成对抗网络的思想就是通过输入条件数据，来约束模型生成的数据的模式．输入的条件数据可以是类别标签，也可以是训练数据的一部分，又甚至是不同模式的数据．CGAN的中心思想是希望可以控制 GAN 生成的图片，而不是单纯的随机生成图片。

- 参考实现：

  ```
  url=https://github.com/znxlwm/pytorch-generative-model-collections
  commit_id=0d183bb5ea2fbe069e1c6806c4a9a1fd8e81656f
  code_path=CGAN.py
  model_name=CGAN
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FP32 | 100 x 72 | HW        |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | 100 x 3 x 28 x 28 | NCHW         |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.5.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   CGAN使用的是随机数作为生成网络的输入。


2. 数据预处理，将随机数转换为模型输入的数据。

   执行CGAN_preprocess.py脚本，完成预处理。

   ```
   python3  CGAN_preprocess.py --save_path=./prep_dataset
   ```
   参数说明：`--save_path`指定数据保存路径。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       权重文件为：[CGAN_G.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/CGAN/PTH/CGAN_G.pth)  
       将获取的权重文件放在当前工作路径下。

   2. 导出onnx文件。

      1. 使用CGAN_pth2onnx.py导出onnx文件。

         运行CGAN_pth2onnx.py脚本。

         ```
         python3 CGAN_pth2onnx.py --pth_path=./CGAN_G.pth --onnx_path=./CGAN.onnx
         ```
         
         参数说明：`--pth_path`表示pth路径，`--onnx_path`表示保存的onnx模型路径。
         获得CGAN.onnx文件。
      
      2. 使用onnx-simplifier优化模型。
        
         ```
          python3 -m onnxsim --input-shape="100,72" CGAN.onnx CGAN_sim.onnx
         ```
         
         参数说明：通过命令`python3 -m onnxsim -h`查看。

         获得CGAN_sim.onnx文件。


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
            --model=CGAN_sim.onnx \
            --output=CGAN_bs1 \
            --input_format=ND \
            --output_type=FP32 \
            --input_shape="image:100,72" \
            --log=error \
            --soc_version=Ascend${chip_name}
         ```

         - 参数说明：通过`atc -h`命令查看。

           运行成功后生成`CGAN_bs1.om`模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        python3 -m ais_bench \
            --model=CGAN_bs1.om \
            --output=./ \
            --outfmt=BIN \
            --input=prep_dataset \
            --batchsize=1
        ```

        -   参数说明：通过命令'python3 -m ais_bench -h'命令查看。

        推理后的输出保存在当前目录下。

   3. 精度验证。

      调用`CGAN_postprocess.py`脚本将om模型的推理结果转化为图像。

      ```
       python3 CGAN_postprocess.py \
            --bin_out_path=${results_path} \
            --save_path=./result
      ```

     - 参数说明：

             -   --bin_out_path：om推理结果路径。
             -   --save_path：为转化图像的生成路径。

      运行后在`--save_path`指定的目录保存转化的图像，生成的图片名称为`result.png`。
     
      精度比对方法：om模型的推理图像`result.png`中包含手写数字效果，与[开源链接](https://github.com/znxlwm/pytorch-generative-model-collections)中展示的在视觉效果上一致。


   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证om模型的性能，参考命令如下：

        ```
         python3 -m ais_bench \
            --model=CGAN_bs1.om \
            --loop=20 \
            --batchsize=1
        ```

      - 参数说明：通过命令'python3 -m ais_bench -h'命令查看。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|  Ascend310 | 1                 | 随机生成数据           | 图片视觉评价           | 1935.7336 fps                |