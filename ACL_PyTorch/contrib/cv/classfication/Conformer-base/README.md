# Conformer-base模型-推理指导


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

Conformer是一个结合卷积和Transformer的混合网络结构，以利用卷积运算和自我注意机制来增强表示学习。Conformer源于特征耦合单元（FCU），它以交互方式融合了不同分辨率下的局部特征和全局表示。Conformer采用并行结构，以便最大程度地保留局部特征和全局表示。



- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmclassification
  code_path=configs/conformer
  model_name=Conformer
  ```
  





## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


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
  | CANN                                                         | 5.1.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.9.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/open-mmlab/mmclassification
   cd mmclassification
   git checkout v0.24.1
   git apply ../Conformer-base.patch
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集http://www.image-net.org/download-images，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。目录结构如下：

   ```
   ├── datasets
      ├── val
      ├── val_label.txt 
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   Conformer-base模型使用ImageNet2012中的5万张验证集数据进行测试，具体来说参考Conformer-base的源码仓中的测试过程对验证集图像进行缩放，中心裁剪以及归一化，并将图像数据转换为二进制文件(.bin) 

   ```
   mkdir ./bin
   python3 Conformer-base_preprocess.py --src_path {dataset_path}/val --save_path ./bin 
   ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      ```bash
      wget https://download.openmmlab.com/mmclassification/v0/conformer/conformer-base-p16_3rdparty_8xb128_in1k_20211206-bfdf8637.pth
      ``` 

   2. 导出onnx文件。

      1. 执行pytorch2onnx.py脚本，生成动态onnx模型文件。

         ```bash
         cd mmclassification
         python3 tools/deployment/pytorch2onnx.py configs/conformer/conformer-base-p16_8xb128_in1k.py --checkpoint ../conformer-base-p16_3rdparty_8xb128_in1k_20211206-bfdf8637.pth --output-file ../conformer_base_dynamicbs.onnx --dynamic-export
         cd ..
         ```

         获得conformer_base_dynamicbs.onnx文件。

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
          atc --model=conformer_base_dynamicbs.onnx --framework=5 --output=conformer_base_bs${batch_size} --input_shape="input:${batch_size},3,224,224" --log=error --soc_version=${soc_version}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --insert\_op\_conf=aipp\_resnet34.config:  AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。

           运行成功后生成<u>***conformer_base_bs${batch_size}.om***</u>模型文件。

2. 开始推理验证

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

        ```
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        mkdir output
        python3 ais_infer.py --model ./conformer_base_bs1.om --input ./bin --output ./output --outfmt TXT  
        ```

        -   参数说明：

             -   model：om文件路径。
             -   input：模型需要的输入。
             -   output：推理结果输出路径。
             -   outfmt：输出数据的格式。

        >**说明：** 
        >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见--help命令。

   3. 精度验证。

      调用Conformer-base_postprocess.py脚本与数据集标签val_label.txt比对，可以获得Accuracy数据，结果保存在acc_result.json中。

      ```
       python3 Conformer-base_postprocess.py --anno_file={dataset_path}/val_label.txt --benchmark_out=output/{ais_infer_out} --result_file=acc_result.json
      ```

      - 参数说明：

        - {dataset_path}/val_label.txt：数据集标签文件


        - output/{ais_infer_out}：生成推理结果所在目录


        - acc_result.json：生成结果文件

   4. 性能验证

      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 ${ais_infer_path}/ais_infer.py --model=conformer_base_bs${batch_size}.om --loop=20 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：对应batch size的om文件
        - --batchsize：batch大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 模型                | 参考精度     | 310p离线推理精度   | 性能基准         | 310p性能        |
|---------------------|-------------|-------------     |--------------   |--------------|
| Conformer-base bs1  | top1:83.82% | top1:83.85%     | fps 151.8395     | fps 142.4363 |


|   模型              |  性能基准  |  310P性能  |
| :------:            | :--------: | :--------: |
| Conformer-base bs1  | 151.8395fps | 142.4363fps |
| Conformer-base bs4  | 211.3495fps | 208.2684fps |
| Conformer-base bs8  | 218.5971fps | 257.7437fps |
| Conformer-base bs16 | 221.4235ps | 231.9296ps  |
| Conformer-base bs32 | 222.7179fps | 220.9309fps |
| Conformer-base bs64 | 138.7254fps | 208.0021fps |
