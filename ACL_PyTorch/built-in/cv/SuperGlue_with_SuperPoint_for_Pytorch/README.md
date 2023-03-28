#   SuperPoint+SuperGlue模型-推理指导

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

SuperGlue网络用于特征匹配与外点剔除，其使用图神经网络对兴趣点进行特征增强，并将特征匹配问题转换为可微分最优化转移问题进行求解。本文档中，SuperGlue基于SuperPoint提取的关键特征点和对应描述子进行图像匹配。

- 参考论文：

  - [SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629)
  - [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/abs/1911.11763)

- 参考实现：

  ```
  https://github.com/magicleap/SuperGluePretrainedNetwork/tree/master/models
  ```

## 输入输出数据<a name="section540883920406"></a>

- SuperPoint 输入数据

  | 输入数据 | 数据类型 | 大小                | 数据排布格式 |
  | -------- | -------- | ------------------- | ------------ |
  | image    | FLOAT32  | 1 x 1 x 1200 x 1600 | NCHW         |

- SuperPoint 输出数据

  | 输出数据    | 数据类型 | 大小             | 数据排布格式 |
  | ----------- | -------- | ---------------- | ------------ |
  | keypoints   | FLOAT32  | points_num x 2   | ND           |
  | scores      | FLOAT32  | points_num       | ND           |
  | descriptors | FLOAT32  | 256 x points_num | ND           |

- SuperGlue 输入数据

  | 输入数据     | 数据类型 | 大小              | 数据排布格式 |
  | ------------ | -------- | ----------------- | ------------ |
  | keypoints0   | FLOAT32  | points_num0 x 2   | ND           |
  | scores0      | FLOAT32  | points_num0       | ND           |
  | descriptors0 | FLOAT32  | 256 x points_num0 | ND           |
  | keypoints1   | FLOAT32  | points_num1 x 2   | ND           |
  | scores1      | FLOAT32  | points_num1       | ND           |
  | descriptors1 | FLOAT32  | 256 x points_num1 | ND           |

- SuperGlue 输出数据

  | 输出数据         | 数据类型 | 大小            | 数据排布格式 |
  | ---------------- | -------- | --------------- | ------------ |
  | matches0         | FLOAT32  | points_num0 x 1 | ND           |
  | matches1         | FLOAT32  | points_num1 x 1 | ND           |
  | matching_scores0 | FLOAT32  | points_num0 x 1 | ND           |
  | matching_scores1 | FLOAT32  | points_num1 x 1 | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1** 版本配套表

  | 配套                                                         | 版本   | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17 | [Pytorch框架推理环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fpies) |
  | CANN                                                         | 6.3RC1 | -                                                            |
  | Python                                                       | 3.7.5  | -                                                            |
  | PyTorch                                                      | 1.12.0 | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \      | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取开源模型代码。

   ```
   git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
   cd SuperGluePretrainedNetwork
   git reset --hard ddcf11f42e7e0732a0c4607648f9448ea8d73590
   ```

2. 获取本仓源码，置于开源模型代码目录下，目录结构如下所示：

   ```
   SuperGluePretrainedNetwork/
   |-- LICENSE
   |-- README.md        # 推理指导文件
   |-- SuperGlue.patch  # 补丁文件
   |-- assets
   |   |-- ......
   |   |-- yfcc_test_pairs_with_gt.txt  # 数据集真值文件
   |   `-- yfcc_test_pairs_with_gt_original.txt
   |-- ......
   |-- infer.py  # 端到端推理脚本
   |-- models
   |   |-- ......
   |   |-- matching.py
   |   |-- superglue.py   # superglue 网络结构
   |   |-- superpoint.py  # superpoint 网络结构
   |   |-- utils.py
   |   `-- weights
   |       |-- superglue_indoor.pth   # superglue 室内场景权重文件
   |       |-- superglue_outdoor.pth  # superglue 室外场景权重文件
   |       `-- superpoint_v1.pth      # superpoint 权重文件
   |-- requirements.txt               # 依赖列表
   |-- superglue_pth2onnx.py          # superglue onnx导出脚本
   `-- superpoint_pth2onnx.py         # superpoint onnx导出脚本
   ```

3. 对源码进行修改。

   ```
   patch -p1 < SuperGlue.patch
   ```

4. 安装必要依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   该模型使用室外场景数据集[YFCC100M](http://projects.dfki.uni-kl.de/yfcc100m/)进行测试。使用如下命令下载解压数据集并置于指定路径：

   ```
   git clone https://github.com/zjhthu/OANet
   cd OANet
   git reset --hard 51d71ff3f57161e912ec72420cd91cf7db64ab74
   bash download_data.sh raw_data raw_data_yfcc.tar.gz 0 8
   tar -xvf raw_data_yfcc.tar.gz
   mv raw_data/yfcc100m ~/data  
   ```

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      已包含在开源代码仓中，详见[获取源码](#section4622531142816)。

   2. 导出onnx文件。

      - 使用superpoint_pth2onnx.py导出SuperPoint的onnx文件。

         ```shell
         python3 superpoint_pth2onnx.py --output_file superpoint.onnx --image_size 1600 1200
         ```

         参数说明：

         - --output_file：导出的onnx文件名。
         - --image_size：输入图像的大小（宽度和高度）。

      - 使用superglue_pth2onnx.py导出SuperGlue的onnx文件。

        ```shell
        python3 superglue_pth2onnx.py --output_file superglue.onnx --superglue outdoor --image_size 1600 1200
        ```

        参数说明：

        - --output_file：导出的onnx文件名。
        - --superpoint：选择加载 indoor/outdoor 场景的权重。
        - --image_size：输入图像的大小（宽度和高度）。

   3. 简化onnx文件。

      ```python
      # superglue
      python3 -m onnxsim superglue.onnx superglue_sim.onnx --dynamic-input-shape --input-shape keypoints0:1,40,2 scores0:1,40 descriptors0:1,256,40 keypoints1:1,40,2 scores1:1,40 descriptors1:1,256,40
      ```
      
   4. 使用ATC工具将ONNX模型转OM模型。
   
      1. 配置环境变量。
   
         ```shell
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
   
   5. 执行ATC命令。
   
      ```shell
      # superpoint
      atc --model superpoint.onnx --output superpoint --framework 5 --log=error --soc_version Ascend310P3
      
      # superglue
      atc --model superglue_sim.onnx --output superglue --framework 5 --log=error --soc_version Ascend310P3 --input_shape_range "keypoints0:[1,-1,2];scores0:[1,-1];descriptors0:[1,256,-1];keypoints1:[1,-1,2];scores1:[1,-1];descriptors1:[1,256,-1]"
      ```
   
      参数说明：
   
      - --model：为ONNX模型文件。
      - --output：输出的OM模型。
      - --framework：5代表ONNX模型。
      - --log：日志级别。
      - --soc_version：处理器型号。
      - --input_shape_range：指定动态输入的输入范围。
   
2. 开始推理验证。

   1. 调用ais-bench工具的推理接口进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

   2. 执行推理。

      ```shell
      python3 infer.py \
      --superpoint_path superpoint.om \
      --superglue_path superglue.om \
      --input_dir ~/data/yfcc100m \
      --input_pairs assets/yfcc_test_pairs_with_gt.txt \
      --resize 1600 1200
      ```
      
      该步骤包括数据集前后处理和模型推理过程，并输出精度和性能结果。
      
      参数说明：
      
      - --superpoint_path：SuperPoint 离线推理文件路径。
      - --superglue_path：SuperGlue 离线推理文件路径。
      - --input_dir：数据集文件路径。
      - --input_pairs：真值标签文件路径。
      - --resize：与输入图像的大小保持一致（宽度和高度）。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

SuperPoint + SuperGlue 模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | pth精度（ AUC@20 \| Prec ） | NPU精度（ AUC@20 \| Prec ) | E2E耗时 |
| ----------- | ---------- | -------- | --------------------------- | -------------------------- | ------- |
| Ascend310P3 | 1          | YFCC100M | 75.08 \| 98.55              | 75.04 \| 97.85             | 5.75s   |
