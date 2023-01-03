# STDC模型-推理指导


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

STDC网络包含了一种新的结构，称为短期密集连接模块(Short-Term density Concatenate module, STDC模块)，以获得具有少量参数的可变感受野；STDC模块被集成到U-net架构中，形成STDC网络，大大提高了网络在语义分割任务中的性能。STDC网络将来自多个连续层的特征图连接起来，每个层对不同尺度和各自领域的输入图像/特征进行编码，从而实现多尺度特征表示。为了加快速度，层的卷积核尺寸逐渐减小，分割性能的损失可以忽略不计。


- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmsegmentation/tree/master/configs/stdc 
  commit_id=43b4efb122f1c4e934ee2588f40210e8c34eed5f
  code_path=ACL_PyTorch/contrib/cv/segmentation/STDC
  model_name=STDC
  ```
  





## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- | -------- | --------------------------- | ------------ |
  | input    | int8     | batchsize x 3 x 1024 x 2048 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小        | 数据排布格式 |
  | -------- | -------- | ----------- | ------------ |
  | output1  | uint32   | 1 x 2097152 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.11.0  | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

- 该模型需要以下依赖   

  **表 2**  依赖列表

  | 依赖名称    | 版本   |
  | ----------- | ------ |
  | onnx        | 1.7.0  |
  | Torch       | 1.11.0 |
  | mmcv-full   | 1.5.3  |
  | TorchVision | 0.6.0  |
  | numpy       | 1.18.5 |
  | Pillow      | 7.2.0  |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/open-mmlab/mmsegmentation.git
   cd mmsegmentation
   pip3 install -e .
   pip install mmcv-full==1.7.0 -f https://download.openmmlcv/dist/cu102/torch1.11/index.html
   pip3 install torchvision==0.12.0
   cd ..
   ```
   
2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   a.本模型支持Cityscapes 500张图片的验证集。请用户需自行获取gtFinet和leftImg8bit数据集（ 注册Cityscapes后下载[gtFine_trainvaltest.zip]( https://www.cityscapes-dataset.com/file-handling/?packageID=1 )和[leftImg8bit_trainvaltest.zip]( https://www.cityscapes-dataset.com/file-handling/?packageID=3 ) ），上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到leftImg8bit验证集及gtFine中的数据标签。目录结构如下：

   ```
   ├── cityscapes
     ├── leftImg8bit
     ├── gtFine
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   将原始数据（.jpeg）转化为二进制文件（.bin）。

   执行STDC_preprocess.py 脚本，完成预处理。

   ```
   python3 ./STDC_preprocess.py /opt/npu/cityscapes/leftImg8bit/val/ ./prep_dataset
   ```

   + 参数说明：
     + 第一个参数：原始数据验证集（.jpeg）所在路径。
     + 第二个参数：输出的二进制文件（.bin）所在路径。

   每个图像对应生成一个二进制文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       从源码包中获取权重文件[stdc1_512x1024_80k_cityscapes_20220224_073048-74e6920a.pth](https://download.openmmlab.com/mmsegmentation/v0.5/stdc/stdc1_512x1024_80k_cityscapes/stdc1_512x1024_80k_cityscapes_20220224_073048-74e6920a.pth)。

       ```
       wget -P ./mmsegmentation/  https://download.openmmlab.com/mmsegmentation/v0.5/stdc/stdc1_512x1024_80k_cityscapes/stdc1_512x1024_80k_cityscapes_20220224_073048-74e6920a.pth
       ```

   2. 导出onnx文件。

      1. 使用mmsegmentation/tools目录下的pytorch2onnx.py导出onnx文件。

         运行mmsegmentation/tools/pytorch2onnx.py脚本。

         ```
         cd mmsegmentation 
         python3 tools/pytorch2onnx.py \
         configs/stdc/stdc1_512x1024_80k_cityscapes.py \
         --checkpoint stdc1_512x1024_80k_cityscapes_20220224_073048-74e6920a.pth \
         --output-file ../stdc_bs1.onnx \
         --cfg-options model.test_cfg.mode="whole"
         ```
         
         获得stdc_bs1.onnx文件。
      
      2. 优化ONNX文件。
      
         在STDC目录下，使用optimize_onnx.py脚本优化onnx模型
      
         ```
         python3 optimize_onnx.py stdc_bs1.onnx stdc_optimize_bs1.onnx
         ```
      
          获得stdc_optimize_bs1.onnx文件。
      
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
         atc --framework=5 --model=./stdc_optimize_bs1.onnx --output=stdc_optimize_bs1 --input_format=NCHW --input_shape="input:1,3,1024,2048" --log=debug --soc_version={soc_version} --insert_op_conf=./aipp.config --enable_small_channel=1
         ```
         
         - 参数说明：
         
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --insert_op_conf：表示插入aipp算子的配置文件。
           -   --enable_small_channel：模型优化参数。
           
         
         运行成功后生成stdc_optimize_bs1.om模型文件。
   
2. 开始推理验证。

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

        ```
        python3 ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model ./stdc_optimize_bs1.om --input ./prep_dataset --output ./result --output_dirname bs1 --outfmt BIN --batchsize 1
        ```
        
        + 参数说明：
          + --model：om模型。
          +  --input：预处理数据集路径。
          +  --output：推理结果所在路径。
          +  --outfmt：推理结果文件格式。
          + --batchsize：不同的batchsize。
        
        推理后的输出默认在当前目录lcmout下。
        
        >**说明：** 
        >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。
        
   3. 精度验证。
   
        调用stdc_postprocess.py脚本与数据集cityscapes/gtFine/val/中的标签比对，可以获得Accuracy数据，结果保存在postprocess_result.txt中。 
   
        ```
        python3 ./STDC_postprocess.py --output_path=./result/bs1 --gt_path=/opt/npu/cityscapes/gtFine/val
        ```
   
        + 参数说明：
          + --output_path：推理结果保存的路径。
          + --gt_path：cityscapes验证集的路径。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

精度对比：

| Model     | STDC         |
| --------- | ------------ |
| 开源精度  | mIoU = 71.82 |
| 310P3精度 | mIoU = 71.81 |

性能对比：

| 芯片型号 | Batch Size   | 数据集 | 性能 |
| --------- | ---------------- | ---------- | --------------- |
| 310P3 | 1 | Cityscapes | 27.95 |