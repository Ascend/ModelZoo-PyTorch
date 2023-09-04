#   T2vec模型-推理指导

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

t2vec是一种基于深度表征学习的轨迹相似性计算方法，通过学习轨迹的表示向量来缓解轨迹数据中不一致采样率和噪声的影响。

- 参考论文：

   [Deep Representation Learning for Trajectory Similarity Computation](https://ieeexplore.ieee.org/document/8509283)

- 参考实现：

  ```
  url=https://github.com/boathit/t2vec
  commit_id=942d2faca5c9b79d806f458524bb197e7516751
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小          | 数据排布格式 |
  | -------- | -------- | ------------- | ------------ |
  | src      | INT64    | seq_len x 256 | ND           |
  | lengths  | INT64    | 1 x 256       | ND           |

- 输出数据

  | 输出数据 | 数据类型 | 大小          | 数据排布格式 |
  | -------- | -------- | ------------- | ------------ |
  | h        | FLOAT32  | 3 x 256 x 256 | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1** 版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fpies) |
  | CANN                                                         | 6.0.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.11.0  | -                                                            |
  | Julia                                                        | 1.6.1   |                                                              |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取开源模型代码。

   ```
   git clone https://github.com/boathit/t2vec
   cd t2vec
   git reset --hard 942d2faca5c9b79d806f458524bb197e75167514
   ```

2. 获取本仓源码，置于t2vec目录下。

3. 对源码进行修改，以正确导出onnx。

   ```
   git apply t2vec.patch
   ```

4. 安装python相关依赖。

   ```shell
   pip3 install -r requirements.txt
   ```

5. 安装julia及其相关依赖。

   ```shell
   # 安装julia
   wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.1-linux-x86_64.tar.gz # 根据实际架构获取
   tar xvfz julia-1.6.1-linux-x86_64.tar.gz
   sudo ln -s `realpath ./julia-1.6.1/bin/julia` /usr/local/bin/julia # 根据实际情况建立软链
   
   # 安装必要依赖
   julia pkg-install.jl # 确保此步在t2vec目录下
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   该模型使用Proto数据集进行测试。该步骤会生成预处理步骤所需要的h5文件和参数文件：

   ```
   wget http://archive.ics.uci.edu/ml/machine-learning-databases/00339/train.csv.zip
   unzip train.csv.zip
   mv train.csv data/porto.csv
   cd preprocessing
   julia porto2h5.jl --datapath ../data
   julia preprocess.jl --datapath ../data
   cd ..
   ```
   
3. 执行预处理脚本，生成数据集预处理后的npy文件

   ```
   julia t2vec_preprocess.jl --datapath ./data --save_dir ./prep_data
   ```

   参数说明：

   - --datapath：数据集所在路径。
   - --save_dir：预处理后数据的保存路径。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      点击下载[权重文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/T2VEC/PTH/best_model.pt)，放置在t2vec目录下。

   2. 导出onnx文件。

      1. 使用t2vect_pth2onnx.py导出动态batch的onnx文件。

         ```
         python3 t2vec_pth2onnx.py -checkpoint best_model.pt -output_file t2vec.onnx -t2vec_batch 256 -prep_data ./prep_data
         ```

         参数说明：

         - --checkpoint：模型权重文件路径。
         - --output_file：ONNX模型文件保存路径。
         - --t2vec_batch：每次解码轨迹的数量
         - --prep_data：预处理得到的数据，即“准备数据集”步骤中的 `--save_dir` 。

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
      atc --model t2vec.onnx \
      --output t2vec \
      --input_shape "src:-1,256;lengths:1,256" \
      --input_format ND \
      --dynamic_dims "60;70;80;90;100" \
      --framework 5 \
      --log=error \
      --soc_version Ascend${chip_name} 
      ```
      
      运行成功后生成 `t2vec.om` 模型文件。
      
      参数说明：
      
      - --model：为ONNX模型文件。
      - --output：输出的OM模型。
      - --input_shape：输入维度。
      - --input_format：输入数据格式。
      - --framework：5代表ONNX模型。
      - --log：日志级别。
      - --soc_version：处理器型号。

2. 开始推理验证。

   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

   2. 执行推理。

      ```
      python3 -m ais_bench --model=t2vec.om  --input prep_data/src,prep_data/lengths \
      --output ./ --output_dirname result --outfmt NPY --auto_set_dymdims_mode 1
      ```
      
      参数说明：
      
      - --model：om模型路径。
      - --input：输入数据所在路径。
      - --output：推理结果输出路径。
      - --output_dirname：推理结果输出子文件夹。
      - --outfmt：推理结果输出格式
      - --auto_set_dymdims_mode：开启根据输入数据自动设置动态维度的模式
   
3. 精度验证。

   执行后处理脚本获得精度数据。

   ```
   julia t2vec_postprocess.jl --datapath ./data --result_dir ./result
   ```

   参数说明：

   - --datapath：数据集所在路径。
   - --result_dir：生成推理结果所在路径，比如此处为result。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

t2vec 模型在数据集 Proto 上的性能和精度参考下列数据。

| 芯片型号    | 数据集 | dbsize | pth精度（mean rank） | npu精度（mean rank) |
| ----------- | ------ | ------ | -------------------- | ------------------- |
| Ascend310P3 | Proto  | 20000  | 2.386                | 2.386               |
| ...         | ...    | 40000  | 3.621                | 3.621               |
| ...         | ...    | 60000  | 5.031                | 5.031               |
| ...         | ...    | 80000  | 6.722                | 6.724               |
| ...         | ...    | 100000 | 8.101                | 8.103               |

| 芯片型号    | 数据集 | 参考平均耗时（ms） | npu平均耗时（ms） |
| ----------- | ------ | ------------------ | ----------------- |
| Ascend310P3 | Proto  | 11.5~12            | 9.85              |
