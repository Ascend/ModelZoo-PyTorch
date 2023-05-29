# CenterNet模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

  - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

CenterNet 是在 2019 年提出的用于目标检测的模型，相比传统依靠 anchors的检测网络，CenterNet 是一种 anchor-free 的目标检测网络，其输出主要为heatmap，获取该热力图分布的中心点即作为目标的中心点。而目标的其他输出，如尺寸和偏移量等则通过在特征图中通过回归得到，这种方法原理简单，兼容性强，在速度和精度上都比较有优势。


- 论文：

  [Objects as Points: Xingyi Zhou, Dequan Wang, Philipp Krähenbühl.(2019)](https://arxiv.org/abs/1904.07850)

- 参考实现：

  ```
  url= https://github.com/xingyizhou/CenterNet 
  branch=master 
  commit_id=2b7692c377c6686fb35e473dac2de6105eed62c6
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 512 x 512 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小                       | 数据排布格式 |
  | -------- | -------- | -------------------------- | ------------ |
  | output1  | FLOAT32  | batchsize x 80 x 128 x  128 | ND           |
  | output2  | FLOAT32  | batchsize x 2 x 128 x  128 | ND           |
  | output3  | FLOAT32  | batchsize x 2 x 128 x  128 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套       | 版本    | 环境准备指导                                                 |
  | ---------- | ------- | ------------------------------------------------------------ |
  | 固件与驱动 | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN       | 5.1.RC2 | -                                                            |
  | Python     | 3.7.5   | -                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 安装依赖。
   ```
   pip install -r requirements.txt
   ```
   
2. 获取源码并安装。

   ```
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    git clone https://github.com/xingyizhou/CenterNet
    cd CenterNet/src/lib/models/networks
    rm -r DCNv2
    rm -r pose_dla_dcn.py
    git clone https://github.com/jinfagang/DCNv2_latest.git
    mv DCNv2_latest DCNv2
    cd DCNv2
    rm -r dcn_v2.py
    cd ../../../../../../
    mv dcn_v2.py CenterNet/src/lib/models/networks/DCNv2
    mv pose_dla_dcn.py DCNv2.patch CenterNet/src/lib/models/networks

    cd CenterNet/src/lib/external
    make
    cd ../models/networks/DCNv2
    python setup.py build develop
    cd ../../../../../../
    export PYTHONPATH=./CenterNet/src/:$PYTHONPATH

   ```

3. 在编译可变形卷积的时候可能出现编译不成功的情况，如果出现下面这类错误，通过打补丁的形式修改相应文件。

   ```
   error: ‘TORCH_CHECK_ARG’ was not declared in this scope
   error: command '/usr/bin/g++' failed with exit code 1
   ```

   1）cd到CenterNet/src/lib/models/networks目录下，执行以下命令打补丁

    ```
    patch -p0 < DCNv2.patch
    ```

   2） 最后再重新执行python setup.py build develop进行编译，即可成功

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   获取COCO数据集：coco2017，下载其中val2017图片及其标注文件（2017 Val images，2017 Train/Val annotations），放入CenterNet/data/coco/路径下，val2017目录存放coco数据集的验证集图片，“annotations”目录存放coco数据集的“instances_val2017.json”。目录结构如下：

   ```
   CenterNet
   ├── data
   │   ├── coco
   │   │   ├── annotations
   │   │   ├── val2017
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行CenterNet_preprocess.py脚本，完成预处理。

   ```
   python CenterNet_preprocess.py data/coco/val2017 prep_dataset
   ```

   参数说明：

   - data/coco/val2017:  原始数据验证集所在路径。
   - prep_dataset:   输出的二进制文件保存路径。

   运行成功后，生成“prep_dataset”文件夹，prep_dataset目录下生成的是供模型推理的bin文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。放在当前目录下 [ctdet_coco_dla_2x.pth](https://pan.baidu.com/s/1e8TIeBvWzb15UEHWCDZcSQ )
       提取码：d446

   2. 导出onnx文件。

      使用ctdet_coco_dla_2x.pth导出onnx文件。

      在CenterNet根目录下，运行CenterNet_pth2onnx.py脚本。

      ```
      python CenterNet_pth2onnx.py ctdet_coco_dla_2x.pth CenterNet.onnx
      ```

      获得CenterNet.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         source /etc/profile
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
         atc --framework=5 --model=CenterNet.onnx  --output=CenterNet_bs1 --input_format=NCHW --input_shape="actual_input:1,3,512,512" --out_nodes="Conv_1120:0;Conv_1123:0;Conv_1126:0" --log=info --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成 CenterNet_bs1.om 模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      ```
      python -m ais_bench --model CenterNet_bs1.om --input prep_dataset --output result --output_dirname dumpout_bs1 --batchsize 1
      ```

      -   参数说明：

           -   --model：om模型的路径。
           -   --input：输入模型的二进制文件路径。
           -   --output：推理结果输出目录。
           -    --output_dirname：推理结果输出的二级目录名。
           -   --batchsize：输入数据的batchsize。

      推理后的输出在当前目录result下。


   3. 精度验证。

      在CenterNet根目录下，执行脚本CenterNet_postprocess_s1.py

      ```
      mkdir save
      python CenterNet_postprocess_s1.py --bin_data_path=./result/dumpout_bs1/  --dataset=./data
      ```
      然后执行执行脚本CenterNet_postprocess_s2.py 获得模型精度信息
      ```
      python CenterNet_postprocess_s2.py --dataset=./data
      ```

      - 参数说明：

        - --bin_data_path：推理结果文件路径
        - --dataset: 原始数据集路径

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
      python -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型的路径
        - --batchsize：数据集batch_size的大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集   | 精度 | 性能    |
| -------- | ---------- | -------- | ---- | ------- |
| 310P3    | 1          | COCO2017 | 36.4 | 32.1448 |
| 310P3    | 4          | COCO2017 | - | 34.1512 |
| 310P3    | 8          | COCO2017 | - | 33.0343 |
| 310P3    | 16         | COCO2017 | - | 33.6273 |
| 310P3    | 32         | COCO2017 | - | 31.8843 |

备注：

1.原官网pth精度 AP : 37.4 是在线推理时keep_res(保持分辨率)的结果，但由于离线推理需要固定shape，故需要去掉keep_res(保持分辨率)。去掉keep_res(保持分辨率)后，跑在线推理精度评估得到 AP : 36.6 ，故以 AP : 36.6 作为精度基准

2.onnx因包含npu自定义算子dcnv2而不能推理，故使用在线推理测试性能

3.原模型在线推理中仅实现batchsize=1的精度测试和性能测试

