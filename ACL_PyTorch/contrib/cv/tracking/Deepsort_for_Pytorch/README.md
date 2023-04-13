# Deepsort模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)
  -  [输入输出数据](#section540883920406)
- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Deepsort是一种多目标跟踪方法，简单有效。该方法将外观信息集成起来，提高了分拣性能，能够在较长遮挡时间下仍能进行有效的跟踪。该框架将大量的复杂计算放入离线预训练阶段，这个阶段在重识别数据集上学习一个深度关联度量。在线应用阶段，建立度量，在视觉外观空间中使用最近邻查询跟踪关联。本模型能够在较快帧率下实现较高精度的识别。


- 参考论文：

  [Simple Online and Realtime Tracking with a Deep Association Metric Nicolai Wojke, Alex Bewley, Dietrich Paulus](https://arxiv.org/abs/1703.07402)

- 参考实现：

  ```
  url=https://github.com/ZQPei/deep_sort_pytorch
  branch=master
  commit_id=4c2d86229b0b69316af67d519f8476eee69c9b20
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 416 x 416 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小              | 数据排布格式 |
  | -------- | -------- | ----------------- | ------------ |
  | output1  | FLOAT32  | 1 x 255 x 52 x 52 | NCHW          |
  | output2  | FLOAT32  | 1 x 255 x 26 x 26 | NCHW          |
  | output3  | FLOAT32  | 1 x 255 x 13 x 13 | NCHW          |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动 

  **表 1**  版本配套表

  | 配套       | 版本                               | 环境准备指导                                                 |
  | ---------- | ---------------------------------- | ------------------------------------------------------------ |
  | 固件与驱动 | 22.0.4（NPU驱动固件版本为6.3.RC1） | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN       | 6.3.RC1                            | -                                                            |
  | Python     | 3.7.5                              | -                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/ZQPei/deep_sort_pytorch.git
   cd deep_sort_pytorch
   git reset 4c2d86229b0b69316af67d519f8476eee69c9b20 --hard
   cd ..
   mv yolov3_deepsort_eval.py deep_sort_pytorch/
   cp acl_net_dynamic.py deep_sort_pytorch/detector/YOLOv3/
   mv acl_net_dynamic.py export_deep_onnx.py deep_sort_pytorch/deep_sort/deep/
   ```

2. 修改开源仓文件

   执行以下命令

   ```
   cd deep_sort_pytorch
   patch -p1 < ../yolov3.patch
   cd ..
   ```

3. 安装依赖。

   ```
   pip install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   该模型使用MOT16训练集的部分进行测试。用户需自行获取数据集，将数据集放到源码路径下，下载 [demo.avi](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6) 并将其移动到deep_sort_pytorch目录下。目录结构如下：

   ```
   deep_sort_pytorch
      └── MOT16
          ├---train
            └──MOT16-02
            └──MOT16-04
            └──MOT16-05
            └──MOT16-09
            └──MOT16-10
            └──MOT16-11
            └──MOT16-13
      └──demo.avi
   ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      使用以下命令下载权重文件

      ```
      wget -P deep_sort_pytorch/detector/YOLOv3/weight/ https://pjreddie.com/media/files/yolov3.weights
      
      wget -P deep_sort_pytorch/deep_sort/deep/checkpoint https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6 
      ```

   2. 导出onnx文件。

      1. 使用yolov3_deepsort.py导出onnx文件。

         在deep_sort_pytorch目录下，运行yolov3_deepsort.py脚本。

         ```
         python3 yolov3_deepsort.py demo.avi --cpu
         ```

         - 参数说明：
           - demo.avi：样本视频路径。
           - --cpu：在cpu下完成onnx的转换。

         获得yolov3.onnx文件。

      2. 该段网络导出的onnx包含动态shape算子where，当前无法支持，因此使用onnxsim进行常量折叠，消除动态shape，在./deep_sort_pytorch/目录下运行

         ```
         pip install onnx-simplifier
         python -m onnxsim yolov3.onnx yolov3-sim.onnx
         ```

         - 参数说明：
           - yolov3.onnx：待简化的onnx模型路径。
           - yolov3-sim.onnx：简化之后的onnx路径。

         获得yolov3-sim.onnx文件。

      3. 将export_deep_onnx.py脚本置于deep_sort_pytorch/deep_sort/deep/路径下，进入该并运行该脚本，导出第二段onnx。

         ```
         cd deep_sort/deep/
         python export_deep_onnx.py
         mv deep.onnx ../../
         cd ../../
         ```

         获得deep.onnx文件。

         > 说明：运行后出现cannot unpack non-iterable NoneType object 错误不影响onnx文件的导出，可忽略该错误。

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
         atc --framework=5 --model=yolov3-sim.onnx --output=yolov3-sim --input_format=NCHW --input_shape="actual_input_1:1,3,416,416" --log=info --soc_version=Ascend${chip_name}
         
         atc --model=deep.onnx --framework=5  --output=deep_dims --input_format=ND  --input_shape="actual_input_1:-1,3,128,64" --dynamic_dims="1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20;21;22;23;24;25;26;27;28;29;30;31;32;33;34;35;36;37;38;39;40;41;42;43;44;45;46;47;48;49;50" --log=error --soc_version=Ascend${chip_name}
         ```

         - 参数说明：
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --dynamic_dims：设置ND格式下动态维度的档位。适用于执行推理时，每次处理任意维度的场景。

         运行成功后生成 <u>***yolov3-sim.om***</u> 和 <u>**deep_dims.om**</u> 模型文件。

     

2. 开始推理验证。

   1. 在Deepsort_for_Pytorch目录下，执行以下命令，删除转onnx的代码并修改evaluation文件。

      ```
      cd ..
      patch -p0 < evaluation.patch
      cd deep_sort_pytorch
      ```

   2. 精度验证。

      将acl_net_dynamic.py脚本放置在detector/YOLOv3/以及deep_sort/deep目录下，将yolov3-sim.om和deep_dims.om以及我们提供的yolov3_deepsort_eval.py放在deep_sort_pytorch目录下，调用脚本yolov3_deepsort_eval.py获取精度。

      ```
      python3 yolov3_deepsort_eval.py --data_root=./MOT16/train --device=0
      ```
	- 参数说明：
	   -  --data_root 数据集路径
           -  --device 芯片id
   3. 性能验证。

      1. 安装ais_bench推理工具。

         请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

      2. 可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

         ```
         python -m ais_bench --model=yolov3-sim.om --loop=20 --batchsize=1
         python -m ais_bench --model=deep_dims.om --loop=20 --dymDims=actual_input_1:1,3,128,64 --batchsize=1
         ```

         - 参数说明：
            - --model：om模型的路径。
            - --batchsize：数据集的batchsize。
            - --dymDims: 动态维度参数，指定模型输入的实际shape。
            - --loop：循环的次数。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

  调用ACL接口推理计算，精度与性能参考下列数据。

  | 芯片型号 | Batch Size | 数据集   | 精度 | 性能    |
  | -------- | ---------- | -------- | ------ | ------- |
  | 310P3    | 1          | MOT16 | 30.0 |  yolov3：464.29；deep：2950 |
