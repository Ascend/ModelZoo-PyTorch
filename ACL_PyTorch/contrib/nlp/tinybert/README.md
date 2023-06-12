# TinyBERT模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

   - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&性能](#ZH-CN_TOPIC_0000001172201573)

  ******

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

TinyBERT是一种新型的Transformer蒸馏方法，该方法能将大型教师BERT模型中的大量知识有效地萃取到小型学生BERT模型中，在加速推理和减少模型大小的同时保持准确性。

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据  | 数据类型  | 大小                      | 数据排布格式  |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 4 x 84 x 84   | NCHW         |


- 输出数据

  | 输出数据  | 数据类型  | 大小               | 数据排布格式  |
  | -------- | -------- | ------------------ | ------------ |
  | output1  | FLOAT32  | batchsize x 4 x 51 | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套        | 版本    | 环境准备指导             |
| ---------- | ------- | ----------------------- |
| 固件与驱动  | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN       | 6.0.0   | -                       |
| Python     | 3.7.5   | -                       |
| PyTorch    | 1.8.0   | -                       |  

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/huawei-noah/Pretrained-Language-Model
   ln -s Pretrained-Language-Model/TinyBERT/transformer/ .
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型将使用到SST-2验证集的dev.tsv文件，通过[链接](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip)下载，获取成功后重命名为glue_dir/SST-2，放到当前工作目录即可。

2. 数据预处理。

   ```
   bash ./test/preprocess_data.sh
   ```

   输出：input_ids, segment_ids, input_mask三个文件夹各放置872笔数据对应的二进制数据文件。
   
## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      从[TinyBERT说明](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)上获取权重文件包“SST-2_model.zip”，解压至当前目录。
       
   2. 导出onnx文件。

         pth权重文件转onnx，并对onnx进行简化。

         ```
         bash ./test/pth2onnx.sh ${bs}
         ```

         注：bs代表批大小。

         输出：若执行bash ./test/pth2onnx.sh 1，则生成TinyBERT_bs1.onnx和TinyBERT_sim_bs1.onnx。

   3. 使用ATC工具将ONNX模型转OM模型。

      ```
      bash ./test/onnx2om.sh ${bs} ${chip_name}
      ```

      注：bs代表批大小；chip_name代表处理器型号。

      输出：若执行bash ./test/onnx2om.sh 1 310P3，则生成TinyBERT_bs1.om。

2. 开始推理验证。
   
   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2.  执行推理。

         ```
         bash ./test/ais_inference.sh ${bs}
         ```

         注：bs代表批大小。

         输出：若执行bash ./test/ais_inference.sh 1，则在当前路径的result文件夹内生成一个新的文件夹，同时在屏幕上打印出性能数据。

   3.  精度验证。

         ```
         bash ./test/postprocess_data.sh ${filename}
         ```

         注：filename代表步骤2中新生成文件夹的名字。

         输出：若步骤2中在/result路径下新生成的文件夹名为ais_infer_result_bs1，执行命令bash ./test/postprocess_data.sh ais_infer_result_bs1,则会在屏幕上打印出精度数据。

         将TinyBERT_sim.onnx上传至T4服务器，测试onnx性能。
      
         ```
         trtexec --onnx=TinyBERT_sim_bs${bs}.onnx --workspace=5000 --threads
         ```
         输出：得到GPU下的推理性能。
   
   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model TinyBERT_bs1.om --loop 1000 --batchsize 1
      ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

以下给出以ais_bench作为推理工具的精度及性能数据：

|<center>模型|<center>官网pth精度|<center>310推理精度|<center>310P推理精度|<center>310性能|<center>310P性能|<center>T4性能|<center>310P/310|<center>310P/T4
|  ----  | ----  | ----|---- |---- | ---- | ---- | ---- | ---- | 
|<center>TinyBERT(bs1)|<center>无|<center>92.66|<center>92.32|<center>707.89|<center>1324.52|<center>972.16|<center>1.87|<center>1.36
|<center>TinyBERT(bs4)|<center>无|<center>92.66|<center>92.32|<center>2047.71|<center>3521.31|<center>2850.36|<center>1.72|<center>1.24
|<center>TinyBERT(bs8)|<center>无|<center>92.66|<center>92.32|<center>2883.62|<center>5871.86|<center>3325.62|<center>2.04|<center>1.77
|<center>TinyBERT(bs16)|<center>无|<center>92.66|<center>92.32|<center>3775.02|<center>8659.63|<center>3415.3590|<center>2.29|<center>2.54
|<center>TinyBERT(bs32)|<center>92.6|<center>92.66|<center>92.32|<center>4301.24|<center>10523.35|<center>3746.7130|<center>2.45|<center>2.81
|<center>TinyBERT(bs64)|<center>无|<center>92.66|<center>92.32|<center>4018.88|<center>11160.38|<center>4425.89|<center>2.78|<center>2.52
|<center>最优bs|<center>92.6|<center>92.66|<center>92.32|<center>4301.24|<center>11160.38|<center>4425.89|<center>2.59|<center>2.52

备注：

- 该模型不支持动态shape

- 性能单位：fps/card，精度为百分比

- bs指batch_size