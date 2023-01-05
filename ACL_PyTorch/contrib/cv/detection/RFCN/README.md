# RFCN模型-推理指导


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
[RFCN论文](https://arxiv.org/abs/1605.06409) 
RFCN基于faster rcnn的基础上对roi pooling这部分进行了改进，与之前的基于区域的检测器相比，此模型的基于区域的检测器是完全卷积的，几乎所有计算都在整个图像上共享，为了实现这一目标，提出了位置敏感得分图来解决图像分类中的平移不变性和目标检测中的平移可变性之间的困境。



- 参考实现：

  ```
  url=https://github.com/RebornL/RFCN-pytorch.1.0
  commit_id=e32e6db63f13c7f27c42bb3a9c447d42cc0b81e4
  model_name=ACL_PyTorch/contrib/cv/detection/RFCN
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | 1 x 3 x 1344 x 1344 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | 1 x 300 x 21 | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.6.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/RebornL/RFCN-pytorch.1.0
   cd RFCN-pytorch.1.0
   git reset --hard e32e6db63f13c7f27c42bb3a9c447d42cc0b81e4
   cd ..
   ```

2. 安装依赖

   ```
   cd RFCN-pytorch.1.0
   pip3 install -r requirements.txt
   conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
   ```
3. 编译
   ```
   cd lib
   python setup.py build develop
   cd ..
   ```
4. 在RFCN-pytorch.1.0目录下创建data文件夹
   ```
   cd RFCN-pytorch.1.0
   mkdir data
   ```
5. coco
   ```
   cd data
   git clone https://github.com/pdollar/coco.git 
   cd coco/PythonAPI
   make
   cd ../../..
   ```
## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持VOCtest_06-Nov-2007验证集。用户需自行获取数据集，然后解压放入data文件夹下

   ```
   RFCN-pytorch.1.0/data/VOCdevkit2007/VOC2007/
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行预处理脚本，生成数据集预处理后的bin文件(第一个参数是输入的图片路径，第二个参数是输出之后的bin文件存放路径)

   ```
   python rfcn_preprocess.py --file_path ./RFCN-pytorch.1.0/data/VOCdevkit2007/VOC2007/JPEGImages/ --bin_path ./bin
   ```

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. RFCN预训练pth权重文件：[faster_rcnn_2_12_5010.pth]

      链接：https://pan.baidu.com/s/1HcGAXmDTLbVm_m5M1T3A3A 
      提取码：er9jg

   2. 导出onnx文件。

      1. 执行pth2onnx脚本，生成onnx模型文件

         ```
         python rfcn_pth2onnx.py  --input ./faster_rcnn_2_12_5010.pth --output rfcn_1.onnx
         ```

         生成rfcn_1.onnx文件

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
          atc --framework=5 --model=./rfcn_1.onnx --output=rfcn_bs1 --input_format=NCHW --input_shape="im_data:1,3,1344,1344" --log=error --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成rfcn_bs1.om模型文件。

2. 开始推理验证

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
         python3 -m ais_bench --model rfcn_bs1.om --input ./bin --output ./ --output_dirname result 
        ```

        -   参数说明：

             -   --model：om模型。
             -   --input：输入数据。
             -   --output：结果输出路径。
             -   --output_dirname: 结果输出文件夹

            推理后的输出默认在当前目录result下。


   3. 精度验证。

      执行rfcn_postprocess.py获得精度

      ```
       python rfcn_postprocess.py --input ./result --output out
      ```


   4. 性能验证

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3.7 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型
        - --batchsize：模型batchsize
        - --loop: 循环次数



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度 | 性能 |
| -------- | ---------- | ------ | ---- | ---- |
|   310P3       |    1        |   coco     |  0.6993    |   16.52   |