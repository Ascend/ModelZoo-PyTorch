# BMN模型-推理指导


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

边界匹配网络（Boundary-Matching Network, BMN）针对边界敏感网络（Boundary Sensitive Network, BSN）方法中所存在的一些短板进行了改进。在时序动作提名生成（temporal action proposal generation）任务中，BMN 能够高效地同时给密集分布的大量时序动作提名生成高质量的置信度分数，在算法效率和算法效果上均有明显提升。

- 参考论文：

  [BMN: Boundary-Matching Network for Temporal Action Proposal Generation](https://arxiv.org/abs/1907.09702)


- 参考实现：

  ```
  url=https://github.com/JJBOY/BMN-Boundary-Matching-Network.git
  commit_id=a92c1d79c19d88b1d57b5abfae5a0be33f3002eb
  ```
  

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                   | 数据排布格式 |
  | -------- | -------- | ---------------------- | ------------ |
  | image    | FP32     | batch_size x 400 x 100 | ND           |


- 输出数据

  | 输出数据       | 数据类型 | 大小                       | 数据排布格式 |
  | -------------- | -------- | -------------------------- | ------------ |
  | confidence_map | FP32     | batch_size x 2 x 100 x 100 | ND           |
  | start          | FP32     | batch_size x 100           | ND           |
  | end            | FP32     | batch_size x 100           | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.12.1  | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取本仓源码。

2. 在同级目录下，获取第三方开源代码仓。

   ```
   git clone https://github.com/JJBOY/BMN-Boundary-Matching-Network.git
   cd BMN-Boundary-Matching-Network
   git reset --hard a92c1d79c19d88b1d57b5abfae5a0be33f3002eb
   ```

3. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型使用处理后的[Activity1.3](https://github.com/wzmsltw/BSN-boundary-sensitive-network#download-datasets)数据集进行推理测试 ，用户自行获取数据集后，将文件解压并上传数据集到指定路径下。

   ```shell
   # 若从 BaiduYun 链接下载，需运行此步
   # zip -FF zip_csv_mean_100.zip --out csv_mean_100.zip
   
   # 解压并上传至指定路径
   unzip csv_mean_100.zip
   mv csv_mean_100 ${path-to-BMN-Boundary-Matching-Network}/data/activitynet_feature_cuhk
   ```
   
   数据集目录结构如下所示：
   
   ```
   data
   |-- activitynet_annotations
   |   |-- action_name.csv
   |   |-- anet_anno_action.json
   |   `-- video_info_new.csv
   `-- activitynet_feature_cuhk
       |-- csv_mean_100
       |   |-- v_---9CpRcKoU.csv
       |   |-- v_--0edUL8zmA.csv
       |   |-- v_--1DO2V4K74.csv
       |   |-- v_--6bJUbfpnQ.csv
   ...
   ```
   
2. 数据预处理，将原始数据集转换为模型的输入数据。

   执行 bmn_preprocess.py 脚本，完成数据预处理。

   ```
   python3 bmn_preprocess.py --save_dir=${save_dir} 
   ```
   参数说明：

   - --save_dir：生成数据集二进制文件的所在路径。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       从开源仓获取权重文件[BMN_AUC67.7.pth.tar](https://pan.baidu.com/s/1ctIV83-Oz9P3jWD1iYnR2g) （提取码：nk3h）

   2. 导出onnx文件。

      1. 使用bmn_tar2onnx.py分别导出**静态**batch的onnx文件。

         ```
         python3 bmn_tar2onnx.py --input_file=${tar_file} --output_file=${onnx_file} --infer_batch_size=${bs}
         ```

         参数说明：

         - --input_file：权重文件。
         - --output_file：生成 onnx 文件。
         - --infer_batch_size：批次大小。
      
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
          # bs = [1, 4, 8, 16, 32, 64]
          atc --model=${onnx_file} --framework=5 --output=bmn_bs${bs} \
          --input-shape="image:${bs},400,100" --log=error --soc_version=Ascend${chip_name}
      ```
      
         运行成功后生成bmn_bs${bs}.om模型文件。
      
         参数说明：
         - --model：为ONNX模型文件。
         - --framework：5代表ONNX模型。
         - --output：输出的OM模型。
         - --input\_format：输入数据的格式。
         - --input\_shape：输入数据的shape。
         - --log：日志级别。
         - --soc\_version：处理器型号。
      
   
2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

      ```
      mkdir result
      python3 -m ais_bench --model=bmn_bs1.om  --batchsize=1 \
      --input ${save_dir} --output result --output_dirname result_bs1
      ```
      
      参数说明：
      
      -   --model：om模型路径。
      -   --batchsize：批次大小。
      -   --input：输入数据所在路径。
      -   --output：推理结果输出路径。
      -   --output_dirname：推理结果输出子文件夹。
   
3. 精度验证。
  
      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。
   
      ```
    python3 bmn_postprocess.py --result_dir=${result_dir}
    ```

      参数说明：
   
      - --result_dir：推理结果所在路径，例如本文档中应为result/result_bs1。
    
4. 可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
  
      ```
      python3 -m ais_bench --model=bmn_bs${bs}.om --loop=50 --batchsize=${bs}
      ```
      
      参数说明：
      - --model：om模型路径。
      - --batchsize：批次大小。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，BMN模型的性能和精度参考下列数据（备注：batch_size=64的BMN模型超出内存限制）。

| 芯片型号    | Batch Size | 数据集          | 精度指标1（AUC） | 性能（FPS） |
| ----------- | ---------- | --------------- | ---------------- | ----------- |
| Ascend310P3 | 1          | ActivityNet 1.3 | 67.69            | 114.34      |
| Ascend310P3 | 4          | ActivityNet 1.3 | 67.69            | 99.02       |
| Ascend310P3 | 8          | ActivityNet 1.3 | 67.69            | 82.86       |
| Ascend310P3 | 16         | ActivityNet 1.3 | 67.69            | 82.94       |
| Ascend310P3 | 32         | ActivityNet 1.3 | 67.69            | 77.23       |

