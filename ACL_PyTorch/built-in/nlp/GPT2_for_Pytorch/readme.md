# GPT2 模型-推理指导


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

GPT-2 模型只使用了多个Masked Self-Attention和Feed Forward Neural Network，并且由多层单向Transformer的解码器构成，本质上是一个自回归模型。其中自回归的意思是指，每次产生新单词后，将新单词加到原输入句后面，作为新的输入句。而单向是指只会考虑在待预测词位置左侧的词对待预测词的影响。


- 参考实现：

  ```
  url=https://github.com/Morizeyao/GPT2-Chinese
  commit_id=bbb44651be8361faef35d2a857451d231b5ebe14
  model_name=ACL_PyTorch/built-in/nlp/GPT2_for_Pytorch
  ```

> <font size=4 color=red>说明：所有脚本都在GPT2的仓下运行</font>

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input_ids    | int64 | batchsize x 512 | ND         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output  | FLOAT16  | batchsize x 512 x 21128 | ND           |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.11.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/Morizeyao/GPT2-Chinese       
   cd GPT2-Chinese            
   git reset --hard bbb44651be8361faef35d2a857451d231b5ebe14
   ```

2. 安装依赖。

   ```
   pip3 install -r requirement.txt
   ```

3. 获取配置文件
   
   从[这里](https://pan.baidu.com/s/16x0hfBCekWju75xPeyyRfA#list/path=%2F)下载配置文件,提取码`n3s8`，并把`pytorch_model.bin`放到`model`下面

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持wiki_zh_2019验证集。用户需自行获取[数据集](https://pan.baidu.com/share/init?surl=22sax9QujO8SUdV3jH5mTQ)，提取码`xv7e`。将解压后的数据放在data下，其目录结构如下：

   ```
   data     
   └── wiki_zh
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   1. 执行`get_data_json.py`脚本，获得原始语料

      ```
      python3 get_data_json.py data/wiki_zh eval.json
      ```
      结果保存在`eval.json`
   2. 执行`pre_data.py`脚本，完成预处理
      
      ```
      python3 pre_data.py
      ```
      结果保存在`data/tokenized_eval`

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。


   1. 导出onnx文件。

      1. 使用`pth2onnx.py`导出onnx文件。
         运行`pth2onnx.py`脚本。

         ```
         python3 pth2onnx.py
         ```

         获得`gpt2_dybs.onnx`文件。

      2. 优化ONNX文件。

         1. 执行onnxsim
            ```
            mkdir onnx_sim_and_modify

            python3 -m onnxsim --overwrite-input-shape=input_ids:4,512 gpt2_dybs.onnx onnx_sim_and_modify/gpt2_4bs_sim.onnx
            ```

            获得`gpt2_4bs_sim.onnx`文件
            > 注意，需要反复执行上一步操作，直到Slice显示为零，第二次开始的时候需要把gpt2_dybs.onnx换成onnx_sim_and_modify/gpt2_4bs_sim.onnx

         2. 改图
            
            使用auto-optimizer改图，获取安装使用请参考[这里](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)

            ```
            python3 opt_onnx_update.py onnx_sim_and_modify/gpt2_4bs_sim.onnx onnx_sim_and_modify/gpt2_4bs_sim_modify.onnx
            ```
            最终获得`gpt2_4bs_sim_modify.onnx`文件

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
         atc --model=onnx_sim_and_modify/gpt2_4bs_sim_modify.onnx \
             --framework=5 \
             --output=om/gpt2_4bs_sim_modify \
             --input_shape=input_ids:4,512 \
             --input_format=ND \
             --log=error \
             --soc_version=Ascend${chip_name} \
             --op_precision=op_precision.ini \
             --output_type=FP16
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --op_precision:性能模式
           -   --output_type:输出结果的类型

           运行成功后生成`gpt2_4bs_sim_modify.om`模型文件。

2. 开始推理验证。
   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais-bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

   2. 执行推理。

        ```
        python3 compare_loss.py --model_path=om/gpt2_4bs_sim_modify.om --batch_size=4
        ```

        -   参数说明：

             -   model_path：om文件路径。
             -   batch_size：模型bs

        推理后的输出精度结果默认在当前目录`eval_result_npu/result.txt`下,性能数据会打屏显示

        >**说明：** 
        >执行ais-bench工具请选择与运行环境架构相同的命令
   

   3. 性能验证。

      可使用ais-bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 -m ais_bench --model=${om_model_path} --loop=50 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型
        - --batchsize：模型batchsize
        - --loop: 循环次数



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度指标（Loss）| 性能 |
| :------: | :--------: | :----: | :--: | :--: |
|     310P3     |      1      |   wiki_zh_2019     |  16.5   |  149    |
|     310P3     |      4      |   wiki_zh_2019     |   16.5   |  188    |
|     310P3     |      8      |   wiki_zh_2019     |   16.5   |   189   |
|     310P3     |      16      |   wiki_zh_2019     |   16.5   |   189   |
|     310P3     |      32      |   wiki_zh_2019     |   16.5   |   185   |
|     310P3     |      64      |   wiki_zh_2019     |   16.5   |    181  |

> 注：衡量精度的指标为验证集平均交叉熵损失（Cross-Entropy Loss），数值越低越好。