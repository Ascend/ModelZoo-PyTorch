# GloRe模型-推理指导


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

卷积神经网络擅长捕获局部关系，而捕获全局关系效率低下。该网络引入GloRe单元，实现全局推理。该方法主要将一组特征在坐标空间上全局聚合，然后投影到另一个空间，在这个空间推理后用于下游任务，从而实现局部和全局的关系推理。


- 参考实现：

  ```
  url=https://github.com/facebookresearch/GloRe
  commit_id=9c6a7340ebb44a66a3bf1945094fc685fb7b730d
  model_name=contrib/cv/classfication/GloRe
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 8 x 224 x 224 | ND         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 101 | ND           |




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
   git clone https://github.com/facebookresearch/GloRe -b master 
   cd GloRe
   git reset --hard 9c6a7340ebb44a66a3bf1945094fc685fb7b730d
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持[UCF101](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)数据集。用户需自行获取数据集,并放在当前路径，解压后目录如下

   ```
   UCF-101
   ├── ApplyEyeMakeup       
   └── ...             
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行`GloRe_preprocess.py`脚本，完成预处理。

   ```
   python3 GloRe_preprocess.py --data-root ./UCF-101 --save-path bin/bs1 --batch-size 1
   ```

  


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       ```
       wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Glore/PTH/GloRe.pth
       ```

   2. 导出onnx文件。

      1. 使用`GloRe_pth2onnx.py`导出onnx文件。
         运行`GloRe_pth2onnx.py`脚本。

         ```shell
         python3 GloRe_pth2onnx.py GloRe.pth GloRe.onnx
         ```

         获得`GloRe.onnx`文件。

      2. 优化ONNX文件。
         ```
         python3 -m onnxsim --input-shape "1,3,8,224,224"  GloRe.onnx GloRe_1bs_sim.onnx
         ```

         获得`GloRe_1bs_sim.onnx`文件。
         > 说明：所有性能数据都是基于优化后的`onnx`转`om`测出的

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
          atc --framework=5 --model=GloRe_1bs_sim.onnx --output=GloRe_bs1 --input_format=NCHW --input_shape="image:1,3,8,224,224" --log=error --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
          
           运行成功后生成`GloRe_bs1.om`模型文件。

2. 开始推理验证。
   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais-bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

   2. 执行推理。

        ```
        python3 -m ais_bench --model GloRe_bs1.om --input bin/bs1 --output ./ --output_dirname result --outfmt TXT
        ```

        -   参数说明：

             -   model：om模型
             -   input：模型输入数据
             -   output：推理结果保存路径
             -   output_dirname: 推理结果保存文件夹
             -   outfmt：推理结果输出格式
                  	
        推理后的输出默认在当前目录result下。

        >**说明：** 
        >执行ais-bench工具请选择与运行环境架构相同的命令

   3. 精度验证。

      调用脚本与数据集标签bs1_target.json比对，可以获得Accuracy数据，结果保存在res_bs1.json中。

      ```
       python3.7 GloRe_postprocess.py --i result --t bs1_target.json --o res_bs1.json
      ```

      - 参数说明：
        - --i：为生成推理结果  
        - --t：为标签数据
        - --o：为生成结果文件

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
|   310P3       |    1        |   UCF101     |  acc1:92.12%<br>acc5:99.56%    |   82   |
|   310P3       |    4        |   UCF101     |  acc1:92.12%<br>acc5:99.56%    |   85   |
|   310P3       |    8       |   UCF101     |  acc1:92.12%<br>acc5:99.56%    |   83   |
|   310P3       |    16        |   UCF101     |  acc1:92.12%<br>acc5:99.56%    |   82   |
|   310P3       |    32        |   UCF101     |  acc1:92.12%<br>acc5:99.56%    |   83   |
|   310P3       |    64        |   UCF101     |  acc1:92.12%<br>acc5:99.56%    |   82   |