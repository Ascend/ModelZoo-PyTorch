# Flownet2模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

FlowNet提出了第一个基于CNN的光流预测算法，虽然具有快速的计算速度，但是精度依然不及目前最好的传统方法。这在很大程度上限制了FlowNet的应用。FlowNet2.0是FlowNet的增强版，在FlowNet的基础上进行提升，在速度上只付出了很小的代价，使性能大幅度提升，追平了目前领先的传统方法。


- 参考实现：

  ```
  url=https://github.com/NVIDIA/flownet2-pytorch
  commit_id=2e9e010c98931bc7cef3eb063b195f1e0ab470ba
  model_name=flownet2-pytorch
  ```




## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input1    | RGB_FP32 | batchsize x 3 x 448 x 1024 | NCHW         |
  | input2    | RGB_FP32 | batchsize x 3 x 448 x 1024 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 2 x 448 x 1024 | NCHW          |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.9.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |




# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/NVIDIA/flownet2-pytorch.git 
   cd flownet2-pytorch
   git reset --hard 2e9e010c98931bc7cef3eb063b195f1e0ab470ba
   patch -p1 < ../flownet2.patch
   cd ..
   mkdir data
   ```

2. 安装依赖。

   ```
   pip install -r requirements.txt
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

   执行Flownet2_preprocess.py脚本，完成预处理。

   ```
   python Flownet2_preprocess.py --batch_size 1 --dataset ./data/training --output ./data_preprocessed_bs${bs}
   ```



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      在该目录下获取模型权重

   2. 导出onnx文件。

      1. 使用Flownet2_pth2onnx.py导出onnx文件。

         运行Flownet2_pth2onnx.py脚本。

         ```
         python Flownet2_pth2onnx.py --batch_size ${bs} --input_path FlowNet2_checkpoint.pth.tar --out_path flownet2_bs${bs}.onnx
         ```

         获得flownet2_bs${bs}.onnx文件。

      2. 优化ONNX文件，安装[MagicOnnx](https://gitee.com/Ronnie_zheng/MagicONNX.git)。

         ```
         python -m onnxsim flownet2_bs${bs}.onnx flownet2_bs${bs}_sim.onnx
         python fix_onnx.py flownet2_bs${bs}_sim.onnx flownet2_bs${bs}_sim_fix.onnx
         ```

         获得flownet2_bs${bs}_sim_fix.onnx文件。

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
             --model=flownet2_bs${bs}_sim_fix.onnx \
             --output=flownet2_bs${bs} \
             --input_format=NCHW \
             --input_shape="x1:${bs},3,448,1024;x2:${bs},3,448,1024" \
             --log=debug \
             --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***flownet2_bs${bs}.om***</u>模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
      python -m ais_bench\
              --model=flownet2_bs${bs}.om \
              --input=./data_preprocessed_bs${bs}/image1,./data_preprocessed_bs${bs}/image2 \
              --output=./ \
              --output_dirname=./result \
              --batchsize=${batch_size}          
        ```

        -   参数说明：

             -   model：om模型地址
             -   input：预处理数据
             -   output：推理结果保存路径
             -   output_dirname:推理结果保存子目录
         

        推理后的输出保存在当前目录result下。


   3. 精度验证。

      调用Flownet2_postprocess.py脚本计算模型精度

      ```
      python Flownet2_postprocess.py --gt_path ./data_preprocessed_${bs}/gt --output_path ./result/ --batch_size ${bs}
      ```

      - 参数说明：

        - gt_path：标识文件地址


        - output_path：推理输出结果

        - batchsize：batchsize大小

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
      python -m ais_bench --model=flownet2_bs${bs}.om --loop=100 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型路径
        - --batchsize：batchsize大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|     Ascend310P3      |        1          |      MPI-Sintel      |     EPE:2.15       |      14.36 |
|     Ascend310P3      |       4         |      MPI-Sintel      |            |       6.10        |
|     Ascend310P3      |        8          |      MPI-Sintel      |     内存超出       |