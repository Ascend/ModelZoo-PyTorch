# EfficientNet_for_Pytorch模型离线推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

EfficientNet是图像分类网络，在ImageNet上性能优异，并且在常用迁移学习数据集上达到了相当不错的准确率，参数量也大大减少，说明其具备良好的迁移能力，且能够显著提升模型效率。模型内部是通过多个MBConv卷积块实现的。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/pycls
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | --------| -------- | -------- | ------------ |
  | output  | FLOAT32  | batchsize x 1000 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | [CANN推理架构准备](https://www/hiascend.com/software/cann/commercial) |
  | Python                                                       | 3.7.5   | 创建anaconda环境时指定python版本即可，conda create -n ${your_env_name} python==3.7.5 |
  | PyTorch                                                      | 1.11.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```sh
   git clone https://github.com/facebookresearch/pycls
   cd pycls
   git reset f20820e01eef7b9a47b77f13464e3e77c44d5e1f --hard
   cd ..
   ```

2. 安装依赖，测试环境时可能已经安装其中的一些不同版本的库，故手动测试时不推荐使用该命令安装

   ```
   pip3.7 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
   
    该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，以ILSVRC2012为例，上传数据集到服务器任意目录并解压。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的meta.mat数据标签。
    
    数据目录结构参考如下格式：

    ```text
    ├──ILSVRC2012_img_val
    ├──val
    ├──ILSVRC2012_devkit_t12
       ├── data
           └── meta.mat
    ```

2. 数据预处理
   1. 首先运行数据集切分脚本ImageNet_val_split.py切分官方val数据集，形成上述目录结构，
      ```
      python3.7 ImageNet_val_split.py ./val ./ILSVRC2012_devkit_t12
      ```
      - 参数说明：

         -   ./val：下载且未分类的ImageNet的val数据集**绝对路径**（如果需要保留val文件夹请先备份）。
         -   ./ILSVRC2012_devkit_t12：官方提供的deckit文件夹**绝对路径**。

   2. 然后将原始数据集转换为模型输入的数据，执行Efficient-B1_preprocess.py脚本，完成预处理。
      ```
      python3.7 Efficient-B0_preprocess.py ./val ./prep_dataset
      ```
      - 参数说明：

         -   ./val：val数据集**绝对路径**。
         -   ./prep_dataset：保存数据集处理后二进制文件的文件夹**绝对路径**。



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
   运行下一步pth2onnx脚本会自动下载权重文件并转为onnx

   2. 导出onnx文件。

      1. 使用pth2onnx导出onnx文件。

         运行pth2onnx脚本。

         ```
         python3.7 Efficient-B0_pth2onnx.py
         ```

         获得Efficient-b0.onnx文件。


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
         atc --model=Efficient-b0.onnx --framework=5 --input_shape="image:8,3,224,224"--output=b0_bs8 --soc_version=Ascend${chip\_name\}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --output：输出的OM模型。
           -   --soc\_version：处理器型号。
           -   --log：日志级别。

           运行成功后生成<u>***b0_bs8.om***</u>模型文件。

2. 开始推理验证。

   1. 使用ais_bench工具进行推理。

      ais_bench工具获取及使用方式请点击查看[ais_bench推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)

   2. 建立软链接
      将prep_dataset文件夹处理为工具可以输入的格式。
      1. 创建用于保存软链接的文件夹
         ```
         mkdir soft_link
         cd soft_link
         ```
      
      2. 建立软链接（若无法建立，可尝试切换root用户重新建立）
         ```
         find ./prep_dataset/ -name "*.bin" | xargs -i ln -sf {} ./soft_link/
         ```
         -   参数说明：

            -   ./prep_dataset/：必须为prep_dataset的绝对路径。
            -   ./soft_link/：必须为soft_link的绝对路径。


   2. 执行推理。

        ```
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        mkdir outputs
        python3.7 -m ais_bench --model b0_bs8.om --input prep_dataset/ --output ./outputs/ --outfmt=TXT --device 0  
        ```

        -   参数说明：

             -   --model：om文件路径。
             -   --input：数据预处理后保存文件的路径。
             -   --output：输出文件夹路径。
             -   --outfmt：输出格式
             -   --device：NPU的ID

        推理后的输出默认在当前目录参数output创建的输出文件夹下，此处为outputs文件夹。


   3. 精度验证。

      通过将outputs中的结果与val_label进行对比后，输出TOP5的精度保存到result.json文件中

      ```
       python3.7 Efficient-B0_postprocess.py --pre_dir ./outputs/{dir} --data_dir ./val/ --save_file ./result.json
      ```

      - 参数说明：

        -   --pre_dir：为生成推理结果所在路径。
        -   --data_dir：为数据集路径。
        -   result.json：最终精度结果保存文件。

   4. 性能验证。

      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3.7 -m ais_bench --model=b0_bs8.om
        ```

      - 参数说明：
        - --model：om模型名称




# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集|  精度TOP1|精度TOP5 |性能|
| --------- | -----------| ----------| ---------------| ------- |----------------|
| 310P3 |  1       | ImageNet |   75.088 |     91.194    |1245.33      |
| 310P3 |  4       | ImageNet |   75.088 |     91.194     | 2179.84      |
| 310P3 |  8       | ImageNet |   75.088 |     91.194     | 2489.11     |
| 310P3 |  16       | ImageNet |  75.088 |      91.194    | 2394.85      |
| 310P3 |  32       | ImageNet |  75.088 |      91.194     |2416.19      |
| 310P3 |  64       | ImageNet |  75.088 |      91.194      |2278.72      |