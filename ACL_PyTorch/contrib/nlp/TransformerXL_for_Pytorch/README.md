#  Transformer-XL 模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

------

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Transformer-XL是一个自然语言处理框架，在Transformer的基础上提出片段级递归机制(segment-level recurrence mechanism)，引入一个记忆(memory)模块（类似于cache或cell），循环用来建模片段之间的联系。并引入了相对位置编码机制(relative position embedding scheme)，代替绝对位置编码，能够更好地捕获长期依赖关系。

- 参考论文：

  [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](http://arxiv.org/abs/1901.02860)

- 参考实现：

  ```
  url=https://github.com/kimiyoung/transformer-xl
  commit_id=44781ed21dbaec88b280f74d9ae2877f52b492a5
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小          | 数据排布格式 |
  | -------- | -------- | ------------- | ------------ |
  | input    | INT64    | 80 x 1        | ND           |
  | mems     | FLOAT16  | 160 x 1 x 512 | ND           |

- 输出数据

  | 输出数据 | 大小          | 数据类型 | 数据排布格式 |
  | -------- | ------------- | -------- | ------------ |
  | output   | 80 x 204      | FLOAT16  | ND           |
  | mems     | 160 x 1 x 512 | FLOAT16  | ND           |
  

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1** 版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fpies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取开源模型代码。

   ```
   git clone https://github.com/kimiyoung/transformer-xl.git
   cd transformer-xl                                   
   git checkout 44781ed21dbaec88b280f74d9ae2877f52b492a5
   ```

2. 获得本仓源码，覆盖开源仓代码，并打补丁。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cp -r ModelZoo-PyTorch/ACL_PyTorch/contrib/nlp/TransformerXL_for_Pytorch/. pytorch/
   patch -p1 < pytorch/sample.patch
   cd pytorch
   ```

3. 安装必要依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   本模型支持 enwik8 数据集，在工作目录执行下述命令

   ```
   mkdir -p data
   cd data
   mkdir -p enwik8
   cd enwik8
   ```

   请用户需自行获取 [enwik8](http://cs.fit.edu/~mmahoney/compression/enwik8.zip) 数据集压缩包，并上传到 enwik8 文件夹路径下。

   ```
   ├── pytorch
   │    ├── data
   │        ├── enwik8
   │            ├── enwik8.zip
   │    ├── ...
   ```

2. 数据预处理，将原始数据集转换为模型的输入数据。

   执行prep_enwik8.py脚本，完成预处理

   ```
   cd ../../
   python3 prep_enwik8.py
   ```

   从test数据集中抽取一部分进行评测

   ```
   cd data/enwik8
   rm -f cache.pt
   mv test.txt test.txt.bak
   head -1000 test.txt.bak > test.txt
   cd ../../
   ```

   运行成功后，生成如下文件目录

   ```
   ├── pytorch
   │    ├── data
   │        ├── enwik8
   │            ├── enwik8.zip
   │            ├── ...
   │            ├── test.txt
   │    ├── ...
   ```

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      获取权重文件 [model.pt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Transformet-XL/PTH/model.pt)，并将其放入当前工作目录。

   2. 导出onnx文件。

      1. 运行run_enwik8_base.sh脚本导出静态batch的onnx文件（此模型当前只支持batch_size=1）。

         ```
         ./run_enwik8_base.sh onnx --work_dir=./
         ```

         参数说明：

         - --参数1：代表执行的模式，本步骤为onnx。
         - --参数2：权重文件所在的路径。

      2. 使用[onnx-simplifer工具](https://github.com/daquexian/onnx-simplifier#python-version)进行简化。

         ```
         # 安装改图工具
         git clone https://gitee.com/ascend/msadvisor.git
         cd msadvisor/auto-optimizer
         python3 -m pip install .
         cd ../..
         
         # 运行改图脚本
         python3 -m onnxsim model.onnx model_sim.onnx
         ```

         参数说明：
   
         - --参数1：简化前onnx模型文件。
         - --参数2：简化后onnx模型文件。
   
      3. 安装[auto_optimizer工具](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer#%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B)，运行fix_onnx.py脚本优化模型。
   
         ```
         python3 fix_onnx.py -i model_sim.onnx -o model_fix.onnx
         ```

         参数说明：

         - --i/--input_onnx：修改前onnx模型文件。
         - --o/--output_onnx：修改后onnx模型文件。
   
   3. 使用ATC工具将ONNX模型转OM模型。
   
      1. 配置环境变量。
   
         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```
   
      2. 执行命令查看芯片名称（${chip_name}）。
   
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
   
   4. 执行ATC命令。
   
      ```
      bash atc.sh model_fix.onnx model_tsxl Ascend${chip_name}
      ```
      
      运行成功后生成model_tsxl.om模型文件。
      
      参数说明：
   
      - --参数1：输入的onnx文件。
      - --参数2：输出的om文件。
      - --参数3：芯片型号。
   
2. 开始推理验证。

   调用aclruntime的接口进行推理验证，获得性能和精度结果。

   ```
   ./run_enwik8_base.sh om_eval --om_path=./model_tsxl.om  --work_dir=./
   ```

   参数说明：
   
   - --参数1：执行模式。
   - --参数2：om模型路径。
   - --参数3：权重文件路径。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，Transformer-XL模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集 | 基准精度（bpc） | 参考精度（bpc） |
| ----------- | ---------- | ------ | --------------- | --------------- |
| Ascend310P3 | 1          | enwik8 | 1.96636         | 1.96663         |

| 芯片型号    | Batch Size | 参考性能（FPS） |
| ----------- | ---------- | --------------- |
| Ascend310P3 | 1          | 287.02          |