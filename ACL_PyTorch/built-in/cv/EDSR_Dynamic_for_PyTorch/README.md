# EDSR(Dynamic)模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

`EDSR`为一种增强的深度超分辨率网络，其性能超过了当前最先进的 `SR` 方法。 该模型的显着性能改进是通过删除传统残差网络中不必要的模块进行的优化实现的。其提出的方法在基准数据集上显示出SOTA的性能，并赢得了 `NTIRE2017` 超分辨率挑战赛。

- 参考实现：

  ```
  url=https://github.com/sanghyun-son/EDSR-PyTorch
  mode_name=EDSR
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                           | 数据排布格式 |
  | -------- | -------- | -------------------------      | ------------ |
  | image    | FLOAT32 | batchsize x 3 x height x width | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小                           | 数据排布格式 |
  | -------- | -------- | --------                       | ------------ |
  | out      | FLOAT32  | batchsize x 3 x height x width | NCHW         |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动:

  **表 1**  版本配套表

  | 配套                                                            | 版本    | 环境准备指导                                                                                          |
  | ------------------------------------------------------------    | ------- | ------------------------------------------------------------                                          |
  | 固件与驱动                                                      | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 6.1.RC1 | -                                                                                                     |
  | Python                                                          | 3.7.5   | -                                                                                                     |
  | PyTorch                                                         | 1.11.0+ | -                                                                                                     |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git        # 克隆仓库的代码
   git checkout master         # 切换到对应分支
   cd ACL_PyTorch/built-in/cv/EDSR_Dynamic_for_Pytorch              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   git clone https://github.com/sanghyun-son/EDSR-PyTorch
   cd EDSR-PyTorch
   git reset 9d3bb0ec620ea2ac1b5e5e7a32b0133fbba66fd2 --hard
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>

   获取 [benchmark](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) 数据集，解压到对应目录，`benchmark` 数据集包含了多个数据集，这里以 `B100` 数据集作为基准：

   目录结构如下：

   ```shell
   benchmark/
   └── B100
       ├── HR
       └── LR_bicubic
   ```
   执行预处理脚本:

   ```
   mkdir -p data_preprocessed/B100
   python3 edsr_preprocess.py -s benchmark/B100/LR_bicubic/X2/ -d data_preprocessed/B100
   ```

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   获取[EDSR_x2预训练pth权重文件](https://cv.snu.ac.kr/research/EDSR/model_pytorch.tar)

   解压压缩包后获取x2的pt文件，文件名：EDSR_x2.pt

   1. 导出onnx文件。

      1. 使用以下脚本导出onnx文件:

         ```
         mkdir -p models/onnx
         python3 pth2onnx.py --scale 2 --n_resblock 32 --n_feats 256 --res_scale 0.1 --input_path EDSR_x2.pt --out_path models/onnx/EDSR_x2.onnx
         ```

         获得EDSR_x2.onnx文件。

   2. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         回显如下：
         +-------------------|-----------------|------------------------------------------------------+
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
         mkdir -p models/om
         atc --model=models/onnx/EDSR_x2.onnx --framework=5 --output=models/om/EDSR_x2 --input_format=ND --log=debug --soc_version=${chip_name} --input_fp16_nodes="image" --output_type=FP16 --input_shape="image:[1,3,100~1080,100~1920]"
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成EDSR_x2.om模型文件。

2. 开始推理验证。<u>***根据实际推理工具编写***</u>

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        mkdir -p outputs/B100
        python3 -m ais_bench --model models/om/EDSR_x2.om --input data_preprocessed/B100 --output outputs/B100 --outfmt NPY --auto_set_dymshape_mode=True
        ```

        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入文件。
             -   --output：输出目录。
             -   --outfmt: 输出格式。
             -   --auto_set_dymshape_mode: 自动设置动态shape模式。

        推理后的输出默认在当前目录outputs/B100下。
    

   3. 精度验证。

      调用脚本与GT label，可以获得精度数据:

      ```
      python3 edsr_postprocess.py --res ./outputs/B100/${timestamp} --HR benchmark/B100/HR --save_path ./result.json
      ```

      - 参数说明：


        - --res：为生成推结果所在路径。
    
        - --HR：为GT图片路径。
    
        - --save_path: 结果保存路径
        
   4. 性能验证。
   
   动态模型性能测试，以几组核心shape为主，先构造随机数据：
   
    ```shell
    # 构造shape：(240, 320),(480, 640),(720,1280),(1080,1920)
    python3 build_input_data.py
    # 测试不同组shape性能，以240x320为例
    python3 -m ais_bench --model models/om/EDSR_x2.om --input inputs/240_320.bin --dymShape "image:1,3,240,320" --outputSize 888888888 --loop 20
    ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

   精度结果：
   
| 模型         | Pth精度      | NPU离线推理精度 |
| :------:     | :------:     | :------:        |
| EDSR_Dynamic | PSNR: 32.352 | PSNR: 32.352    |

   调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Input shape    | 性能        |
|----------|----------------|-------------|
| 310P3    | H:240, W:320   | 6.9664 fps  |
| 310P3    | H:480, W:640   | 1.6063 fps  |
| 310P3    | H:720, W:1280  | 0.4342 fps  |
| 310P3    | H:1080, W:1920 | 0.1803 fps  |
| 基准性能 | H:240, W:320   | 17.5390 fps |
| 基准性能 | H:480, W:640   | 4.5005 fps  |
| 基准性能 | H:720, W:1280  | 1.5082 fps  |
| 基准性能 | H:1080, W:1920 | 0.6677 fps  |
