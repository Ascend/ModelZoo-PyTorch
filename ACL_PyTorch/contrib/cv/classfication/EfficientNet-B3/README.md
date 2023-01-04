# EfficientNet-B3模型PyTorch离线推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)
   - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)





# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

EfficientNet-B3是一种卷积神经网络，该网络是在EfficientNet-B0的基础上，利用NAS搜索技术，对输入分辨率Resolution、网络深度Layers、网络宽度Channels三者进行综合调整的结果。

- 参考实现：

  ```
  url=https://github.com/rwightman/pytorch-image-models
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FLOAT32 | batchsize x 3 x 300 x 300 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小  | 数据排布格式 |
  | -------- | --------|  ------ | ------------ |
  | output  | FLOAT32  |  batchsize x 1000   | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | [CANN推理架构准备](https://www/hiascend.com/software/cann/commercial) |
  | Python                                                       | 3.7.5   | 创建anaconda环境时指定python版本即可，conda create -n ${your_env_name} python==3.7.5 |
  | PyTorch                                                      | 1.11.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |                                                          |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/facebookresearch/pycls
   cd pycls  
   git reset f20820e01eef7b9a47b77f13464e3e77c44d5e1f --hard  
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
   
    该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，以ILSVRC2012为例，上传数据集到服务器任意目录并解压。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。
    
       ├── ImageNet
         ├── ILSVRC2012_img_val
         ├── val_label.txt

2. 数据预处理。

   数据预处理将原始数据集（.jpeg）转换为模型输入的二进制文件（.bin）。

   执行imagenet_torch_preprocess.py脚本，完成预处理。

    ```
   python3.7 imagenet_torch_preprocess.py efficientnetB3 ./dataset/ImageNet/ILSVRC2012_img_val ./prep_dataset
    ```

     - 参数说明：
       - efficientnetB3：默认输入。
       - ./dataset/ImageNet/ILSVRC2012_img_val：为验证集路径。
       - ./prep_dataset：为预处理后生成的二进制文件的存储路径。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
   
         ```
         wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/EfficientNet-B3/PTH/EN-B3_dds_8gpu.pyth
         ```

   2. 导出onnx文件。

      1. 使用efficientnetB3_pth2onnx.py导出onnx文件。

         运行efficientnetB3_pth2onnx.py脚本。

         ```
         python3.7 efficientnetB3_pth2onnx.py EN-B3_dds_8gpu.pyth ./pycls/configs/dds_baselines/effnet/EN-B3_dds_8gpu.yaml efficientnetB3.onnx
         ```

         获得efficientnetB3.onnx文件。
      
      2. 优化ONNX文件。

         ```
         python3.7 -m onnxsim --overwrite-input-shape="image:16,3,300,300" ./EfficientB3.onnx efficientB3_bs16_onnxsim.onnx
         ```

         获得efficientB3_bs16_onnxsim.onnx文件。

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
         atc --framework=5 --model=./efficientB3_bs16_onnxsim.onnx --input_format=NCHW --input_shape="image:16,3,300,300" --output=efficientnetB3_bs16 --log=debug --soc_version=Ascend${chip_name}  
         ```

         - 参数说明：
           -   --framework：5代表ONNX模型。
           -   --model：为ONNX模型文件。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --output：输出的OM模型。           
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***efficientnetB3_bs16.om***</u>模型文件。



2. 开始推理验证。

   1.  使用ais-infer工具进行推理。

         ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2.  执行推理。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         python3.7 -m ais_bench --model ./efficientnetB3_bs16.om --input ./prep_dataset  --output ./result_bs16 --outfmt TXT   
         ```

         - 参数说明：

            - --model：om文件路径。
            - --input：模型需要的输入(预处理后的生成文件)。
            - --output：为推理数据输出路径。
            - --outfmt：输出数据的格式，可取值“NPY”、“BIN”、“TXT”，精度验证需要TXT格式。  

               推理后的输出在output参数对应路径的文件result_bs16里,推理结果保存在sumary.json里，便于汇总统计。


   3.  精度验证。

         调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

         ```   
         cd result_bs16
         rm -r sumary.json
         cd ..
         python3.7 imagenet_acc_eval.py ./result_bs16 ./val_label.txt ./ result.json
         ```   
         - 参数说明
            - ./result_bs16：为生成推理结果所在路径 
            - ./datasets/imagenet/val_label.txt：为标签数据 
            - result.json：为生成结果文件


   4.  性能验证。

         可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

         ```
         python3.7 -m ais_bench --model=efficientnetB3_bs16.om
         ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集|  精度TOP1 | 精度TOP5 | 性能|
| --------- | ----| ----------| ------     |---------|---------|
| 310P3 |  1       | ImageNet |   76.25     |   92.56  |   482.63      |
| 310P3 |  4       | ImageNet |   76.25     |   92.56  |    571.51      |
| 310P3 |  8       | ImageNet |   76.25     |   92.56  |  579.54     |
| 310P3 |  16       | ImageNet |   76.25     |   92.56  |   666.00      |
| 310P3 |  32       | ImageNet |   76.25     |   92.56  |   555.33      |
| 310P3 |  64       | ImageNet |   76.25     |   92.56  |   556.82      |