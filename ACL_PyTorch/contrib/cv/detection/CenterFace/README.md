# CenterFace模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

  - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

- 论文：[CenterFace: Joint Face Detection and Alignment Using Face as Point](https://arxiv.org/abs/1911.03599)


- 参考实现：

  ```
  url=https://gitee.com/andyrose/center-face.git
  branch=master
  commit_id=063db90e844fa0271abc14067b871f5afcbe6c60
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 800 x 800 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小       | 数据排布格式 |
  | -------- | -------- | ---------- | ------------ |
  | output1  | FLOAT32  | 1 x 40000  | ND           |
  | output2  | FLOAT32  | 1 x 80000  | ND           |
  | output3  | FLOAT32  | 1 x 80000  | ND           |
  | output4  | FLOAT32  | 1 x 400000 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套        | 版本    | 环境准备指导                                                 |
  | ----------- | ------- | ------------------------------------------------------------ |
  | 固件与驱动  | 1.0.17（NPU驱动固件版本为6.0.RC1）  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN        | 6.0.RC1 | -                                                            |
  | Python      | 3.7.5   | -                                                            |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/Levi990223/center-face.git
   ```

2. 整理代码结构

   ```
   mv centerface_pth_preprocess.py centerface_pth_postprocess.py convert.py pth2onnx.py move.sh ./center-face/src
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   获取WIDER_FACE数据集，在center-face目录下创建一个data目录，然后将下载下来的图片数据放在这个data目录下。目录结构如下：

   ```
   center-face
   ├── data
   │   ├── img1
   │   |  ├── img.jpg
   │   ├── img2
   │   |  ├── img.jpg
   │   ├── img3
   │   |  ├── img.jpg
   ```

2. 获取权重文件model_best.pth。放在center-face/src/目录下。
3. 数据预处理，将原始数据集转换为模型输入的数据。

   1. 在center-face/src路径下，执行以下命令编译nms。
      ```
      cd lib/external/
      python setup.py build_ext --inplace
      cd ../../
      ```
   2. 执行centerface_pth_preprocess.py脚本，完成预处理。
      ```
      python centerface_pth_preprocess.py ../data ../after_images/
      ```
      - 参数说明：
         - ../data:  原始数据验证集所在路径。
         - ../after_images/:   输出的二进制文件保存路径。

      运行成功后，生成after_images文件夹，after_images目录下生成的是供模型推理的bin文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件[model_best.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/CenterFace/PTH/model_best.pth)。
   2. 导出onnx文件。

      使用pth2onnx.py导出onnx文件。将pth2onnx.py移动到center-face/src/lib目录下
      
      在center-face/src/lib目录下，运行pth2onnx.py脚本。

      ```
      python pth2onnx.py
      ```

      在目录center-face/src下，获得 CenterFace.onnx 文件。

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

      3. 切换目录到center-face/src下，执行ATC命令。

         ```
         atc --framework=5 --model=CenterFace.onnx --input_format=NCHW --input_shape="image:1,3,800,800" --output=CenterFace_bs1 --log=debug --soc_version=${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成CenterFace_bs1.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      ais_bench推理工具获取及使用方式请点击查看 [ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)

   2. 执行推理。

      ```
      python -m ais_bench --model CenterFace_bs1.om --input ../after_images/ --output result --output_dirname dumpout_bs1 --batchsize 1
      ```

      -   参数说明：

           -   --model：om模型的路径。
           -   --input：输入模型的二进制文件路径。
           -   --output：推理结果输出目录。
           -   --output_dirname：推理结果输出的二级目录名。
           -   --batchsize：输入数据的batchsize。

      推理后的输出在当前目录result下。


   3. 处理目录result/dumpout_bs1下的bin文件，将该目录下的文件分类别存放，以便于后处理。

      在center-face/src目录下，执行convert.py文件

      ```
      mkdir result/result
      python convert.py ./result/dumpout_bs1/ ./result/result
      ```

   4. 精度验证。

      在center-face/src目录下，调用脚本centerface_pth_postprocess.py进行推理结果的后处理。需要将center-face/evaluate/groud_truth路径下的wider_face_val.mat拷贝至center-face/src路径下

      1. 执行后处理脚本

         ```
         python centerface_pth_postprocess.py
         ```

      2. 在center-face/evaluate目录下，执行以下命令编译bbox。
         ```
         python setup.py build_ext --inplace
         ```

      3. 精度验证。在center-face/evaluate目录下，执行evaluation.py文件进行精度验证。

         ```
         python evaluation.py
         ```

   5. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
      python -m ais_bench --model=CenterFace_bs1.om --loop=20 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型的路径
        - --batchsize：数据集batch_size的大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集    | 精度                          | 性能    |
| -------- | ---------- | --------- | ----------------------------- | ------- |
| 310P3    | 1          | widerface | hard：74.55%<br/>easy：92.24%<br/>Medium：91.02% | 439.9085 |
| 310P3    | 4          | widerface | hard：74.55%<br/>easy：92.24%<br/>Medium：91.02% | 412.4094 |
| 310P3    | 8          | widerface | hard：74.55%<br/>easy：92.24%<br/>Medium：91.02% | 375.9275 |
| 310P3    | 16         | widerface | hard：74.55%<br/>easy：92.24%<br/>Medium：91.02% | 369.6435 |
| 310P3    | 32         | widerface | hard：74.55%<br/>easy：92.24%<br/>Medium：91.02% | 371.7701 |



