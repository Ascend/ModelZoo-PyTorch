# C51模型-推理指导


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

C51是一种值分布强化学习算法，C51算法的框架依然是DQN算法，采样过程依然使用epsilon-greedy策略取期望贪婪，并且采用单独的目标网络。与DQN算法不同的是，C51算法的卷积神经网络不再是行为值函数，而是支点处的概率，C51算法的损失函数不再是均方而是KL散度。

- 参考实现：

  ```
  url=https://github.com/ShangtongZhang/DeepRL
  branch=master
  commit_id=13dd18042414ad112bd0bd383a836d8d739e8acf
  model_name=C51
  ``` 
 
  通过Git获取对应commit_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据  | 数据类型  | 大小              | 数据排布格式  |
  | -------- | -------- | ----------------- | ------------ |
  | input    | RGB_FP32 | 1 x 4 x 84 x 84   | NCHW         |


- 输出数据

  | 输出数据  | 数据类型  | 大小       | 数据排布格式  |
  | -------- | -------- | ---------- | ------------ |
  | output   | FLOAT32  | 1 x 4 x 51 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套       |  版本    | 环境准备指导             |
| ---------- | ------- | ----------------------- |
| 固件与驱动  | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN       | 6.0.0   | -                       |
| Python     | 3.7.5   | -                       |
| PyTorch    | 1.8.0   | -                       |  

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/ShangtongZhang/DeepRL
   cd DeepRL
   git apply ../c51-infer-update.patch
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   pip3 install mpi4py
   git clone https://github.com/openai/baselines.git
   cd baselines
   pip3 install -e .
   cd ..
   ```
    >**说明：** pip在线安装requirements.txt中tensorflow==2.6.0仅支持x86架构

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   该模型没有原始输入的数据集，故而将在线推理的输入输出保存作为数据集和标签

2. 数据预处理。

   将在线推理生成的输入输出保存为pt文件，并将输入pt文件转成bin。

   1. 执行“c51_preprocess.py”脚本，完成预处理。

      通过训练获取c51.model权重文件和c51.stats模型配置文件。

      ```
      python3 c51_preprocess.py c51.model c51.stats dataset/states dataset/actions 1000
      ```
      - 参数说明：
   
         - “c51.model”：权重文件

         - “c51.stats”：模型配置文件

         - “dataset/states”：stats输出的二进制文件（.pth）所在路径

         - “dataset/actions”：action输出的二进制文件（.pth）所在路径
   
      运行成功后生成文件：
   
      dataset/states与actions目录下将分别生成stats与action输出的二进制文件
   
   2. 生成数据集bin文件
   
      运行“get_dataset_bin.py”脚本。
 
      ```
      python3 get_dataset_bin.py dataset/states dataset/bin dataset/out
      ```
      - 参数说明：
   
         - “dataset/states”：预处理后的数据文件的相对路径

         - “dataset/bin”：生成的数据集文件保存的路径

         - “dataset/out”：bin文件推理后的保存根目录
   
      运行成功后生成文件：
   
      dataset/bin目录下生成数据集bin文件
   
## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      [c51.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/C51/PTH/c51.model)
       
   2. 导出onnx文件。

         使用c51.model导出onnx文件。

         运行c51_pth2onnx.py脚本。

         ```
         python3 c51_pth2onnx.py --model-path='c51.model' --onnx-path='c51.onnx'
         ```

         获得c51.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         回显如下：
             +--------------------------------------------------------------------------------------------+
             | npu-smi 22.0.0                       Version: 22.0.2                                       |
             +-------------------+-----------------+------------------------------------------------------+
             | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
             | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
             +===================+=================+======================================================+
             | 0       310P3     | OK              | 17.0         56                0    / 0              |
             | 0       0         | 0000:AF:00.0    | 0            934  / 23054                            |
             +===================+=================+======================================================+
         ```

      3. 执行ATC命令。
         ```
         atc --framework=5 --model=c51.onnx --output=c51_bs1 --input_format=NCHW --input_shape="input:1,4,84,84" --log=error --soc_version=${chip_name}  --op_select_implmode=high_performance
         ```

         - 参数说明：

            - --model：为ONNX模型文件
            - --framework：5代表ONNX模型
            - --output：输出的OM模型
            - --input\_format：输入数据的格式
            - --input\_shape：输入数据的shape
            - --log：日志级别
            - --soc\_version：处理器型号
            - --op_select_implmode: 高性能模式

         运行成功后生成c51_bs1.om模型文件。

2. 开始推理验证。
   
   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2.  执行推理。
         ```
         python3 -m ais_bench --model=c51_bs1.om --input dataset/bin  --output dataset/out/ --outfmt TXT --batchsize 1
         ```

         - 参数说明：
         
            - --model：om模型的路径

            - --input：输入的bin文件目录

            - --output：推理结果输出路径
         
            - --outfmt：输出数据的格式

            - --batchsize：模型输入批次大小
   
         说明： 执行ais_bench工具请选择与运行环境架构相同的命令。

   3.  精度验证。

         调用脚本与数据集标签比对，可以获得Accuracy数据。

         ```
         python3 c51_postprocess.py dataset/actions dataset/out/${time_stamp} 1000
         ```

         - 参数说明：

            -   “dataset/actions”：保存的输出action的路径

            -   “dataset/out/${time_stamp}”：离线推理结果的路径

            -  “1000”：参数输出比较的个数
   
   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model c51_bs1.om --loop 1000 --batchsize 1
      ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| batch_size | 310P       |
|------------|------------|
| bs1        | 6050.12fps |

精度参考下列数据。

| 310精度  | 99.6% |
|----------|-------|
| 310P精度 | 98.9% |

注：此模型不支持多batch。
 