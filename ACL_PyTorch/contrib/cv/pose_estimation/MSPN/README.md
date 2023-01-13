# MSPN 模型-推理指导


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

该模型结构使用多阶段网络来实现人体姿势估计，可以使得低分辨率的特征和高分辨率的特征重复交叠，兼顾了位置信息和特征抽象信息。


- 参考实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  commit_id=05360171699bbe3bb1916bbafa84a445bd112e87 
  model_name=ACL_PyTorch/contrib/cv/pose_estimation/MSPN
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 256 x 192 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output  | FLOAT32  | batchsize x 3 x 256 x 192 | NCHW           |




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
   git clone https://github.com/megvii-research/MSPN -b master   
   cd  MSPN 
   git reset --hard 05360171699bbe3bb1916bbafa84a445bd112e87
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```
3. 设置环境变量，将当前目录设置为程序运行的主目录
   ```shell
   export MSPN_HOME=$(pwd)
   export PYTHONPATH=$PYTHONPATH:$MSPN_HOME
   ```
4. 下载COCOAPI

   ```
   git clone https://github.com/cocodataset/cocoapi.git $MSPN_HOME/lib/COCOAPI
   cd $MSPN_HOME/lib/COCOAPI/PythonAPI
   make install
   ```
## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持coco2014验证集。用户需自行获取数据集，将下载好的数据按照如下目录放置：

   ```
   ./dataset
   ├── COCO
      ├── det_json
      ├── gt_json    
      └── images
         ├── train2014
         └── val2014
   ```
2. 运行配置文件
   ```
   python3 $MSPN_HOME/exps/mspn.2xstg.coco/config.py -log
   ```

3. 数据预处理，将原始数据集转换为模型输入的数据。

   执行`MSPN_preprocess.py`脚本，完成预处理。

   ```
   python3 MSPN_preprocess.py --datasets_path ./dataset/COCO
   ```
   - 参数说明
      - datasets_path: 原数据路径
   
   结果默认保存在当前文件夹`pre_dataset`
  


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       ```
       wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/MSPN/PTH/mspn_2xstg_coco.pth
       ```

   2. 导出onnx文件。

      1. 使用`MSPN_pth2onnx.py`导出onnx文件

         运行`MSPN_pth2onnx.py`脚本。

         ```
         python3 MSPN_pth2onnx.py        
         ```
         获得`MSPN.onnx`文件

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

         ```
         atc --framework=5 --model=MSPN.onnx --output=MSPN_bs1 --input_format=NCHW --input_shape="input:1,3,256,192" --log=error --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成`MSPN_bs1.om`模型文件。

2. 开始推理验证。
   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais-bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

   2. 执行推理。

        ```
        python3 -m ais_bench  --model MSPN_bs1.om --input ./pre_dataset --output ./ --output_dirnamne result
        ```

        -   参数说明：

             -   model：om模型
             -   input：输入文件
             -   output：结果输出地址
             -   output_dirname: 结果输出文件夹

        推理后的输出默认在当前目录`result`下。

        >**说明：** 
        >执行ais-bench工具请选择与运行环境架构相同的命令。参数详情请参见。

   3. 精度验证。

      调用脚本，结果会打屏显示

      ```
       python MSPN_postprocess.py --inference_result ./result
      ```

      - 参数说明：

        - inference_result: 推理结果文件


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
|     310P3     |      1      |    COCO2014    |  AP:74.1    |   595   |
|     310P3     |      4     |    COCO2014    |  AP:74.1    |   933   |
|     310P3     |      8      |    COCO2014    |  AP:74.1    |   815   |
|     310P3     |      16      |    COCO2014    |  AP:74.1    |   786   |
|     310P3     |      32      |    COCO2014    |  AP:74.1    |  717    |
|     310P3     |      64      |    COCO2014    |  AP:74.1    |  713    |