# FairMOT 模型-推理指导


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

FairMOT用于目标跟踪，它使用基于CenterNet的方法进行目标检测，然后在训练的过程中进行ReID的过程，提高了跟踪的准确率


- 参考实现：

  ```
  url=https://github.com/ifzhang/FairMOT
  commit_id=2f36e7ebf640313a422cb7f07f93dc53df9b8d12
  model_name=contrib/cv/detection/FairMOT
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 608 x 1088 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 1 x 152 x 272 | NCHW           |
  | output2  | FLOAT32  | batchsize x 4 x 152 x 272 | NCHW           |
  | output3  | FLOAT32  | batchsize x 128 x 152 x 272 | NCHW           |
  | output4  | FLOAT32  | batchsize x 2 x 152 x 272 | NCHW           |





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

   + 安装DCN以及修改DCN代码
   ```shell
   git clone -b pytorch_1.5 https://github.com/ifzhang/DCNv2.git
   cd DCNv2
   python3.7 setup.py build develop
   git reset 9f4254babcd162a809d165fa2430a780d14761f4 --hard
   patch -p1 < ../dcnv2.diff  
   cd ..
   ```

   + 下载并修改开源模型代码

   ```shell
   git clone -b master https://github.com/ifzhang/FairMOT.git
   cd FairMOT
   git reset 2f36e7ebf640313a422cb7f07f93dc53df9b8d12 --hard
   patch -p1 < ../fairmot.diff
   cd ..
   ```


2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
   本模型支持[MOT17](https://motchallenge.net/data/MOT17.zip 'MOT17')数据集。将数据做如下处理

   ```
   mkdir dataset
   cd dataset
   wget https://motchallenge.net/data/MOT17.zip
   unzip MOT17.zip
   cd MOT17
   mkdir images
   mv train/ images/
   mv test/ images/
   cd ../..
   python3 FairMOT/src/gen_labels_16.py
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行`fairmot_preprocess.py`脚本，完成预处理。

   ```
   python3 fairmot_preprocess.py --data_root=./dataset --output_dir=./pre_dataset 
   ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       ```shell
       wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/FairMOT/PTH/fairmot_dla34.pth
       ```

   2. 导出onnx文件。

      1. 使用`fairmot_pth2onnx.py`导出onnx文件。

         运行`fairmot_pth2onnx.py`脚本。

         ```shell
         python3 fairmot_pth2onnx.py --input_file=fairmot_dla34.pth --output_file=fairmot.onnx
         ```

         获得`fairmot.onnx`文件。


   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         会显如下：
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

         ```shell
         atc --framework=5 --model=./fairmot.onnx --input_format=NCHW --input_shape="actual_input_1:1,3,608,1088" --output=./fairmot_bs1 --log=error --soc_version=Ascend${chip_name}
          
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成`fairmot_bs1.om`模型文件。

2. 开始推理验证。
   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais-bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

   2. 执行推理。

        ```
        python3 -m ais_bench  --model fairmot_bs1.om --input ./pre_data --output ./ --output_dirname result --outfmt BIN
        ```

        -   参数说明：

             -   model：om模型
             -   input：输入文件
             -   output：结果输出路径
             -   output_dirname: 结果输出文件夹
             -   outfmt：结果输出格式

        推理后的输出默认在当前目录`result`下。

        >**说明：** 
        >执行ais-bench工具请选择与运行环境架构相同的命令

   3. 精度验证。

      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

      ```
       python3 fairmot_postprocess.py --data_dir=./dataset  --input_root=./result > bs_1_result.log
      ```

      - 参数说明：
        - input_root：为生成推理结果所在路径  
        - data_dir：原始文件路径
      
      结果保存在`bs_1_result.log`

   4. 性能验证。

      可使用ais-bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型
        - --batchsize：模型batchsize
        - --loop: 循环次数



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度 | 性能 |
| -------- | ---------- | ------ | ---- | ---- |
|    310P3      |      1      |    MOT    |  MOTA:83.7    |  11.6    |
|    310P3      |      4      |    MOT    |  MOTA:83.7    |   12   |
|    310P3      |      8      |    MOT    |  MOTA:83.7    |    12  |
|    310P3      |      16      |    MOT    |  MOTA:83.7    |   12   |
|    310P3      |      32      |    MOT    |  MOTA:83.7    |   12   |
