# SSD-ResNet34模型-推理指导


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

SSD模型是用于图像检测的模型，通过基于Resnet34残差卷积网络(基础网络)，并向网络添加辅助结构，产生具有多尺度特征图的预测。在多个尺度的特征图中使用不同的默认框形状，可以有效地离散地输出不同大小的框，面对不同的目标可以有效地检测到，并且还可以对目标进行识别。




- 参考实现：

  ```
  url=https://github.com/mlcommons/training_results_v0.7
  commit_id=elc4b963b6a4ee8fbd40fc5cd9edb9789a2982de
  code_path=contrib/cv/detection/SSD-Resnet34
  model_name=ssd
  ```
  
 


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 300 x 300 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | ploc   | FLOAT32  | Batchsize x 4 x 8732 | ND           |
  | plabel  | FLOAT32       | Batchsize x 81 x 8732 | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
                                               



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/mlcommons/training_results_v0.7.git
   cd training_results_v0.7/NVIDIA/benchmarks/ssd/implementations/pytorch/ 
   patch -p1 <../../../../../../ssd.patch       # 通过补丁修改仓库代码
   mv * ../../../../../../../SSD-Resnet34/      # 移动到模型所在路径      
   ```

2. 安装依赖。
   ```
   pip3 install -r requirements.txt
   git clone https://github.com/mlperf/logging.git mlperf-logging
   pip3 install -e mlperf-logging
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。
   本模型已在coco 2017数据集上验证过精度。推理数据集采用coco_val_2017，请用户自行获取coco_val_2017数据集。将instances_val2017.json文件和val2017文件夹按照如下目录结构上传并解压数据集到服务器任意目录。本模型将使用到coco2017数据集中的验证集及其标签文件instances_val2017.json, bbox_only_instances_val2017.json，标签文件bbox_only_instances_val2017.json是将coco2017中的原标签文件instances_val2017.json经过处理所得。
    最终，数据的目录结构如下：
   ```
   ├── coco
       ├── val2017   
       ├── annotations
            ├──instances_val2017.json
            ├──bbox_only_instances_val2017.json

   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。
   执行prepare-json.py脚本（该脚本在开源仓https://github.com/mlcommons/training_results_v0.7.git training_results_v0.7/NVIDIA/benchmarks/ssd/implementations/pytorch/路径下获取 ），得到bbox_only_instances_val2017.json文件

   ```
   python3 prepare-json.py --keep-keys ${data_path}/coco/annotations/instances_val2017.json ${data_path}/coco/annotations/bbox_only_instances_val2017.json
   ```

   执行ssd_preprocess.py脚本，完成预处理。

   ```
   python3 ssd_preprocess.py --val_annotation=./coco/annotations/bbox_only_instances_val2017.json --data_root=./coco/val2017/ --save_path=./ssd_bin
   ```
   - 参数说明：
      -  --data：数据集路径。
      -  --bin-output：预处理后的数据文件的相对路径。

    
    运行成功后，会在当前目录下生成二进制文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       [iter_183250.pt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/SSD-Resnet34/PTH/iter_183250.pt) 和 [resnet34-333f7ec4.pth](
https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/SSD-Resnet34/PTH/resnet34-333f7ec4.pth)

   2. 导出onnx文件。

      1. 使用ssd_pth2onnx.py导出onnx文件。

        使用“resnet34-333f7ec4.pth”和“iter_183250.pt”导出onnx文件。
        运行“ssd_pth2onnx.py”脚本。

         ```
         python3 ssd_pth2onnx.py --bs=1 --resnet34-model=./models/resnet34-333f7ec4.pth --pth-path=./models/iter_183250.pt --onnx-path=./ssd.onnx
         ```
         - 参数说明：
            -  --resnet34-model : resnet34 骨干网络权重路径
            -  --pth-path: SSD-ResNet34 权重路径
            -  --onnx-path: onnx文件路径

         获得ssd.onnx文件（默认为动态导出）。



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
            atc --framework=5\ 
                 --model=./ssd.onnx\ 
                 --output=./ssd_bs1\ 
                 --input_format=NCHW\ 
                 --input_shape="image:1,3,300,300"\ 
                 --log=error\
                 --soc_version=Ascend${ChipName}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

        运行成功后生成ssd_bs1.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。 
    ```
    python3 -m ais_bench --model ${om_path}/ssd_bs1.om\  
                                  --input /path/to/ssd_bin/\ 
                                  --output ./\ 
                                  --output_dirname result \
                                  --batchsize ${n}\
                                  --outfmt BIN
    ```

    - 参数说明：

      - --model: OM模型路径。
      - --input: 存放预处理bin文件的目录路径
      - --output: 存放推理结果的目录路径
      - --output_dirname：存放推理结果的目录文件名
      - --batchsize：每次输入模型的样本数
      - --outfmt: 推理结果数据的格式

        推理后的输出保存在当前目录result下。


   3. 精度验证。

      调用“ssd_postprocess.py”评测模型的精度。

    ```
    python3 ssd_postprocess.py --val_annotation=./coco/annotations/bbox_only_instances_val2017.json --bin_path=${output_path}
    ```
    - 参数说明：

      - --val_annotation：生成推理结果所在路径。
      - --bin_path：推理结果保存路径
    
   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

    ```
    python3 -m ais_bench --model ${om_path}/ssd_bs1.om --loop 100 --batchsize 1
    ```

    - 参数说明：

      - --model: om模型
      - --batchsize: 每次输入模型样本数
      - --loop: 循环次数    



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

1. 精度对比

    | Model       | batchsize | Accuracy | 
    | ----------- | --------- | -------- |
    | ssd_resnet34 | 1       | map = 23% |
    | ssd_resnet34 | 16     | map = 23% |
2. 性能对比

    | batchsize | 310 性能 | 310P 性能 | 
    | ---- | ---- | ---- |
    | 1 | 711.356  | 960.2 |
    | 4 | 825.516 | 1435.2 |
    | 8 | 849.156 | 1550.3 |
    | 16 | 862.468 | 1503.1 |
    | 32 | 812.368 |1516.5 |
    | 64 | 810.368 | 961.1 |