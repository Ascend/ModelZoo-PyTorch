# Deepctr(WDL、xDeepFM、AutoInt)-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  ******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

DeepCTR的设计主要是面向那些对深度学习以及CTR预测算法感兴趣的人，使他们可以利用这个包：从一个统一视角来看待各个模型; 
快速地进行简单的对比实验;
利用已有的组件快速构建新的模型。

本实验实现了WDL、xDeepFM、AutoInt模型。

- 参考实现：

  ```
  url=https://github.com/shenweichen/DeepCTR
  commit_id=35288ae484d7a32887d6a75bdb48d84db992a892
  model_name=WDL/xDeepFM/Autoint
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 大小     | 数据类型    | 数据排布格式 |
  |--------|---------| -------- | ------------ |
  | input | 40 x 6 | FLOAT16 | ND           |


- 输出数据

  | 输出数据 | 大小     | 数据类型    | 数据排布格式 |
  |--------|---------| -------- | ------------ |
  | output  | 40 x 1 | FLOAT16 | ND           |



# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本      | 环境准备指导                                                 |
| ------------------------------------------------------------ |---------| ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.10.1  | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/shenweichen/DeepCTR.git
   git reset --hard 35288ae484d7a32887d6a75bdb48d84db992a892
   pip3 install -v -e .
   ```

2. 安装依赖。

   ```
   pip3 install --no-deps deepctr-torch
   pip3 install -U deepctr
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   数据集采用代码自带的简单数据集，使用位置位于Deepctr/example/movielens_sample.txt。
   ```
    复制文件到我们的仓下    
    cp /usr/local/DeepCTR/examples/movielens_sample.txt /usr/local/deepctr/
   ```
   
2. 数据预处理。

   数据较简单，省略此步操作。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       框架未开源相关权重文件，由于数据集较简单，可以使用**train_checkpoint.py**自行在gpu环境下训练。
        
       gpu环境下训练同样可以参照上述步骤获取源码和安装依赖，将train_checkpoint.py放于源码Deepctr/example目录下即可
       ```
         python3 train_checkpoint.py --model_name {model_name} --data_name_or_path "./movielens_sample.txt" --device_id {device_id}
      
         示例
         python3 train_checkpoint.py --model_name "WDL" --data_name_or_path "./movielens_sample.txt" --device_id 0
       ```
      - 参数说明：
      
           model_name：需要训练的模型权重名称，可以从三个中任选：xDeepFM,WDL,AutoInt

           data_name_or_path：数据集所在路径
           
           device_id: 使用的GPU号，按实际填写

           运行成功后在当前目录下生成**model_name_weight.h5**权重文件，本例为生成**WDL_weight.h5**。
           
           生成的权重文件同样转移到/usr/local/deepctr路径下

   2. 导出onnx文件。

      1. 使用**export2onnx.py**导出onnx文件。

         运行**export2onnx.py**脚本。

         ```
           mkdir model
           python3 export2onnx.py --model_name {model_name} --data_name_or_path "./movielens_sample.txt"
         
           示例
           python3 export2onnx.py --model_name "WDL" --data_name_or_path "./movielens_sample.txt"
         ```
         > **说明：** 
         运行成功后在model文件夹下生成**model_name.onnx**模型文件，本例为生成**WDL.onnx**。

      2. 使用onnx-simplifier。

         按照链接下载onnx-simplifier[[onnx-simplifier链接](https://github.com/daquexian/onnx-simplifier)]

         ```
         cd model
         python3 -m onnxsim WDL.onnx WDL_sim.onnx
         ```
         > **说明：** 
         按照实际，选择onnx模型，运行成功后在当前目录下生成**model_name_sim.onnx**模型文件，本例为生成**WDL_sim.onnx**。
        
      3. 改图。

         需要修改onnx，以达到提升性能的目的。

         ```
         python3 fix_graph.py --input_model {input_model_path} --output_model {output_model_path}
         
         示例
         python3 fix_graph.py --input_model "./model/WDL_sim.onnx" --output_model "./model/WDL_fix.onnx"
         ```
         - 参数说明：
      
           input_model：需要修改的onnx路径

           output_model：输出保存的onnx路径

           选择onnx模型，运行成功后在model文件夹下生成**model_name_fix.onnx**模型文件，本例为生成**WDL_fix.onnx**。
         
   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

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
         bash deepctr_atc.sh ./model/WDL_fix.onnx ./model/WDL Ascend310P3
         ```

         - 参数说明：
           ./model/WDL_fix.onnx，onnx文件所在路径
         
           ./model/WDL，输出的om文件名称

           Ascend310P3，给soc_version传参数，该参数支持Ascend310和Ascend310P[1-4]

           运行成功后在model文件夹下生成**model_name.om**模型文件，本例为生成**WDL.om**。

2. 开始推理验证。

   a.  使用ais_bench工具进行推理。

      ais_bench工具获取及使用方式请点击查看[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]


   b.  执行推理。

      ```
      python3 infer.py --model_path {input_model_path} --data_name_or_path "./movielens_sample.txt" --device_id {device_id}

      示例
      python3 infer.py --model_path ./model/WDL.om --data_name_or_path "./movielens_sample.txt" --device_id 0
      ```

      -   参数说明：

           -   model_path：需要推理om模型的路径。
           -   data_name_or_path：模型需要的数据集路径。
           -   device_id：推理需要的卡号。
		...

      >**说明：** 
      >执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见。

   c.  精度验证。

      执行推理后自动输出结果。


   d.  性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model ./WDL.om --loop 1000 --batchsize 40

      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号  | model   | Batch Size | 数据集      | 精度     | 性能      |
|-------|---------|------------|----------|--------|---------|
| 310P3 | WDL     | 40         | movielens_sample | 2.1479 | 0.079ms |
| 310P3 | xDeepFM | 40         | movielens_sample | 1.9712 | 0.177ms |
| 310P3 | AutoInt | 40         | movielens_sample | 2.1465 | 0.22ms  |