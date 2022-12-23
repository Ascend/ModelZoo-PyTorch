# Twins_PCPVT_S模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手Twins_PCPVT_S](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  ******

  


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Twins_PCPVT_S使用CPVT中提出的条件位置编码(CPE)来代替PVT中的绝对PE。CPE以输入为条件，可以自然地避免绝对编码的问题；生成CPE的位置编码发生器(PEG)被放置在每个阶段的第一个编码器块之后；对于图像级分类，在CPVT之后，删除类令牌，并在阶段结束时使用全局平均池化(GAP)。Twins-PCPVT继承了PVT和CPVT的优点，易于有效实现。广泛的实验结果表明，这种简单的设计可以匹配最近最先进的Swin transformer的性能。


- 参考实现：

  ```
  url=https://github.com/Meituan-AutoML/Twins
  branch=main
  commit_id=4700293a2d0a91826ab357fc5b9bc1468ae0e987
  model_name=PCPVT-Small
  ```
  


  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 1 x 1000 | FLOAT32  | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.7.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   https://github.com/Meituan-AutoML/Twins.git 
   branch=main 
   commit_id=4700293a2d0a91826ab357fc5b9bc1468ae0e987
   
   cd ./Twins
   git clone https://github.com/Meituan-AutoML/Twins.git 
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   该模型使用[ImageNet2012](https://image-net.org/)

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行twins_pcpvt_s_preprocess.py脚本，完成预处理。

   ```bash
   python3 twins_pcpvt_s_preprocess.py \
       --data_path ${datasets_path} \
       --prep_dataset ./prep_dataset
   ```
   
   参数说明：
   
   ```bash
   data_path: 处理前原数据集的地址
   prep_dataset: 生成数据集的文件夹名称
   ```
   
   运行后，将会得到如下形式的文件夹：
   
   ```bash
   ├── prep_dataset
   │    ├──input_00000.bin
   │    ├──......
   ```
   


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       链接：https://drive.google.com/file/d/1TWIx_8M-4y6UOKtbCgm1v-UVQ-_lYe6X/view

   2. 导出onnx文件。

      1. 使用twins_pth2onnx.py导出onnx文件。

         运行twins_pth2onnx.py脚本。

         ```bash
         python3 twins_pth2onnx.py --source "./pcpvt_small.pth" --target "./twins_dynamic.onnx"
         ```

         获得twins_dynamic.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```bash
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```bash
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
      
         ```bash
         atc --framework=5 --model=./twins_dynamic.onnx \
             --output=./twins_bs${batch_size}_gelu \
             --input_format=NCHW --input_shape="input:${batch_size},3,224,224" \
             --log=debug --soc_version=${chip_name} --op_precision_mode=op_precision.ini
         ```
      
         - 参数说明：
      
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --op_precision_mode:  指定部分算子执行高性能模式 。
      
           运行成功后生成<u>***twins_bs${batch_size}.om***</u>模型文件。
      
   
2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请点击本链接进行安装ais_bench推理工具，以及查看具体使用方法(https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)  

   2. 执行推理。

      ```bash
       python3 -m ais_bench --model ./twins_bs${batch_size}.om --output ./result --outfmt BIN --input ./prep_dataset
      ```

      - 参数说明
        - outfmt：模型类型。
        -  model：om文件路径。
        -  output：结果存放路径。...

        推理后的输出默认在当前目录result下。

      >**说明：** 
      >执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见。

   3. 精度验证。

      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

      ```bash
      python3 twins_pcpvt_s_postprocess.py --folder-davinci-target ./result --annotation-file-path /opt/npu/imageNet/val_label.txt --result-json-path ./result --json-file-name result.json
      ```

       ./result：为生成推理结果所在路径 

        val_label.txt：为标签数据

        result.json：为生成结果文件

   4. 性能验证

       可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```bash
      python3 -m ais_bench --model ./twins_bs${batch_size}.om  --output ./  --outfmt BIN --loop 100 --batchsize=${batch_size}
      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

|  芯片型号   | Batch Size |    数据集    |     精度     |  性能  |
| :---------: | :--------: | :----------: | :----------: | :----: |
| Ascend 310P |     1      | ImageNet2012 | top1：81.22% | 307FPS |
| Ascend 310P |     4      | ImageNet2012 | top1：81.22% | 458FPS |
| Ascend 310P |     8      | ImageNet2012 | top1：81.22% | 602FPS |
| Ascend 310P |     16     | ImageNet2012 | top1：81.22% | 613FPS |
| Ascend 310P |     32     | ImageNet2012 | top1：81.22% | 555FPS |