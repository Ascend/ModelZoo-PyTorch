# Real-ESRGAN模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

  - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Real-ESRGAN 旨在开发通用图像恢复的实用算法。作者将强大的使用纯合成数据进行训练的 ESRGAN 扩展到实际的恢复应用程序（即 Real-ESRGAN）。作者在ESRGAN的基础上进行改进，在以RRDB(residual in residual dense block)为主要模块的ESRGAN的基础上，提出了一种更符合真实世界的退化策略，使用模糊核、噪声、尺寸缩小、压缩四种操作的随机顺序退化图像，并使用新构建的数据集进行训练，最终在真实图像上取得了显著的效果。


- 参考实现：

  ```
  url=https://github.com/xinntao/Real-ESRGAN.git
  branch=master
  commit_id=c9023b3d7a5b711b0505a3e39671e3faab9de1fe
  model_name=Real-ESRGAN
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据
  - 精度测试
    | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
    | ------- | -------- | ------------------------- | ------------ |
    | input.1 | RGB_FP32 | batchsize x 3 x 220 x 220 | NCHW         |

  - 性能测试
    | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
    | ------- | -------- | ------------------------- | ------------ |
    | input.1 | RGB_FP32 | batchsize x 3 x 64 x 64 | NCHW         |

- 输出数据
  | 输出数据 | 数据类型 | 大小                      | 数据排布格式 |
  | ------- | -------- | ------------------------- | ------------ |
  | output | RGB_FP32 | batchsize x 3 x 880 x 880 | NCHW         |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.16（NPU驱动固件版本为5.1.RC2）  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |                                                         |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/xinntao/Real-ESRGAN.git  
   cd Real-ESRGAN 
   git reset c9023b3d7a5b711b0505a3e39671e3faab9de1fe --hard
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

    推理数据集代码仓已提供，并且放置在代码仓./Real-ESRGAN/inputs目录

2. 数据预处理。

    数据预处理将原始数据集转换为模型输入的数据：

    执行`ESRGAN_preprocess.py`脚本，完成预处理。
    - 对于性能测试使用64x64的输入
      ~~~shell
      python3 Real-ESRGAN_preprocess.py ./Real-ESRGAN/inputs/ ./prep_dataset_bin_64 64 64
      ~~~
    - 对于精度测试使用220x220的输入
      ~~~shell
      python3 Real-ESRGAN_preprocess.py ./Real-ESRGAN/inputs/ ./prep_dataset_bin_220 220 220
      ~~~
    - 参数说明：
      -   `./Real-ESRGAN/inputs/` 为输入的图像目录路径
      -   `./prep_dataset_bin` 为输出bin文件路径
      -   `220 220` 输出后的bin文件被裁剪为220x220的大小

    运行成功后会生成名为`prep_dataset_bin_64`和`prep_dataset_bin_220`目录，其中包含裁剪后的图像，并以bin的格式存储。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       将权重文件[RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)放到experiments/pretrained_models/目录
       ```shell
       mkdir -p experiments/pretrained_models
       wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models 
       ```

   2. 导出onnx文件。

      使用Real-ESRGAN_pth2onnx.py导出onnx文件。

        运行./Real-ESRGAN_pth2onnx.py脚本。
        ```shell
        python3 ./Real-ESRGAN_pth2onnx.py --bs=${batch_size} --input_size=220 --onnx_output=./realesrgan-x4-bs${batchsize}.onnx
        ```
       - 参数说明：
          -   --bs: 为输入图像的batch_size，默认为`1`
          -   --input_size 为输入图像尺寸，默认为`220`
          -   --onnx_output 为onnx输出路径，默认为`realesrgan-x4.onnx`
      
        此处，导出onnx模型时需要指定输入的尺寸，若需进行精度推理则使用220x220，若需进行性能推理则使用默认尺寸64x64
        
        最终获得`realesrgan-x4-bs${batchsize}.onnx`文件。


   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

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
          ```shell
          atc --framework=5 --model=realesrgan-x4-b${batchsize}.onnx --output=realesrgan_bs${batchsize} --input_format=NCHW --input_shape="input.1:{batchsize},3,64,64" --log=error --soc_version=Ascend${chip_name}
          ```

          - 参数说明：
            -   --model：为ONNX模型文件。
            -   --framework：5代表ONNX模型。
            -   --output：输出的OM模型。
            -   --input\_format：输入数据的格式。
            -   --input\_shape：输入数据的shape。
            -   --log：日志级别。
            -   --soc\_version：处理器型号。
          
          精度推理需要将`--input_shape`设置为`"input.1:1,3,220,220"`，性能推理需要将`--input_shape`设置为`"input.1:1,3,64,64"`。

           运行成功后生成`realesrgan_bs${batchsize}.om`模型文件。



2. 开始推理验证。

    1. 安装ais_bench推理工具。

       请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。  


    2.  执行推理。
        ```shell
        python3 -m ais_bench --model ./realesrgan_bs${batchsize}.om --batchsize ${batchsize} --output ./result --input ./prep_dataset_bin
        ```
          - 参数说明：
            - --model：om模型路径
            - --batchsize：batchisize大小
            - --output: 输出路径
            - --input：输入bin文件目录

        推理后的输出默认在当前目录result下。
        >**说明：** 
        > 在进行精度推理时，需要指定输入图像的路径并使用输入尺寸为$220 \times 220$的om模型。

    3. 精度验证
           由于代码仓与论文当中并没有提供精度指标，作者是在代码仓的inputs文件夹下进行的推理，生成原图的恢复图像。所以我们参考作者的方法，在npu上进行了图像推理，精度看图像生成的效果。
        在离线推理结束后，使用脚本将输出的BIN文件，进行后处理得到恢复后的图像。生成的图像保存在img_path文件夹下。
        ```shell
        rm -rf ./img_path
        mkdir ./img_path
        python3 Real-ESRGAN_postprocess.py  ./result/dumpOutput_device0  ./img_path
        ```
        - 参数说明：
           - ./result/dumpOutput_device0：为推理生成的bin文件
           - img_path：为生成图像结果文件
      
        将生成图像结果文件，打开并查看模型效果。

    4. 性能验证
        ```shell
        python3 -m ais_bench --model ./realesrgan_bs${batchsize}.om --batchsize ${batchsize} --output ./result --outfmt BIN --loop 5
        ```
        - 参数说明：
          - --model：om模型路径
          - --batchsize：batchisize大小
          - --output: 输出路径
          - --outfmt: 输出文件格式
          - --loop: 循环次数


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。
  1. 性能对比

  | Throughput | 310      | 310P     | T4      | 310P/310    | 310P/T4 |
  | ---------- | -------- | -------  | ------- | ----------- | ------- |
  | bs1        | 140.2496 | 183.9636 | 12.5324 | 1.31        | 14.68   |
  | bs4        | 137.1692 | 251.6356 | 19.7268 | 1.83        | 12.76   |
  | bs8        | 121.4272 | 171.9542 | 21.9474 | 1.42        | 7.83    |
  | bs16       | 117.7756 | 142.1262 | 22.0155 | 1.21        | 6.46    |
  | bs32       | 112.1108 | 135.5650 | 22.6540 | 1.21        | 5.98    |
  |            |          |          |         |             |         |
  | 最优batch  | 140.2496 | 251.6356 | 22.6540 | 1.79        | 11.11   | 
  
  2. 精度对比
  - 原始输入图像

    ![原始输入](https://foruda.gitee.com/images/1661936252729537681/814f6ceb_8600636.png "图片1.png")
    
  - 推理结果图：

    ![推理结果1](https://foruda.gitee.com/images/1661676458669849672/0f13c736_8600636.png "0014_1.png")
    ![推理结果2](https://foruda.gitee.com/images/1661676470063451286/6283e830_8600636.png "0030_1.png")
    ![推理结果3](https://foruda.gitee.com/images/1661676481459202408/226a13fd_8600636.png "ADE_val_00000114_1.png")
    ![推理结果4](https://foruda.gitee.com/images/1661676498632382961/70b3b615_8600636.png "wolf_gray_1.png")
