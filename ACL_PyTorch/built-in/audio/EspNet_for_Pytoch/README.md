# Espnet_conformer模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section183221994400)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能精度](#ZH-CN_TOPIC_0000001172201573)






# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Espnet_conformer模型是一个使用conformer结构的ASR模型。

- 参考实现：

  ```
  url=git clone https://github.com/espnet/espnet
  branch=v.0.10.5
  model_name=tacotron2
  ```
  

通过Git获取对应commit\_id的代码方法如下：

```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
```

EspNet安装比较复杂，请参考https://espnet.github.io/espnet/installation.html

若安装mkl失败，则去launchpad.net/ubuntu/+source/intel-mkl/2020.0.166-1  

下载 intel-mkl_2020.0.166.orig.tar.gz 文件，解压后 bash install.sh安装即可

注意：mkl arm不适用于arm版本安装，推荐适用x86环境

## 输入输出数据<a name="section540883920406"></a>

- encoder输入数据

  | 输入数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input | input_dynamic_axes_1 x 83 | FLOAT32 | ND |


- encoder输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | 2863 | Add2863_dim_0x Add2863_dim_1 x 256 | FLOAT32  | ND |
  




# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.10.1  | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>
安装依赖。

   ```
   pip3 install -r requirements.txt
   ```
   om_gener安装

   ```
   git clone https://gitee.com/peng-ao/om_gener.git
   cd om_gener
   pip3 install .
   ```

   acl_infer安装

   ```
  git clone https://gitee.com/peng-ao/pyacl.git
  cd pyacl
  pip3 install .
   ```

## 获取源码<a name="section183221994400"></a>

在工作目录下执行下述命令获取源码并切换到相应路径。

请按照官方指导文档进行代码的安装

## 准备数据集<a name="section183221994411"></a>

在espnet/egs/aishell/asr1/文件夹下运行bash run.sh --stage -1 –stop_stage -1下载数据集

运行bash run.sh --stage 0 --stop_stage 0处理数据集

运行bash run.sh --stage 1 --stop_stage 1处理数据集

运行bash run.sh --stage 2 --stop_stage 2处理数据集

运行bash run.sh --stage 3 --stop_stage 3处理数据集

若缺少对应的文件夹，则自己建立文件夹

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   本模型基于开源框架PyTorch训练的Espnet_conformer进行模型转换。使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

    1. 在checkpoints目录下获取权重文件。

        下载路径：https://github.com/espnet/espnet/blob/master/egs/aishell/asr1/RESULTS.md
        
        对应Conformer(kernel size = 15) + SpecAugment + LM weight = 0.0下面的model link即可
        
        解压，将对应的conf，data, exp文件夹置于espnet/egs/aishell/asr1
        
   2. 导出onnx文件并转换模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         export ASCEND_GLOBAL_LOG_LEVEL=3
         /usr/local/Ascend/driver/tools/msnpureport -g error -d 0
         ```

         > **说明：** 
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

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

      3. 首先将export_onnx.sh和adaptespnet.py置于espnet/egs/aishell/asr1文件夹下

      ①静态shape

      将export_onnx.diff放在espnet根目录下，

      ```
      git checkout .
      git checkout v.0.10.5
      patch -p1 < export_onnx.diff
      cd ./egs/aishell/asr1/
      bash export_onnx.sh
      ```

      生成encoder.onnx，运行python3.7.5 adaptespnet.py生成encoder_revise.onnx

      ②动态shape

      将export_onnx_dynamic.diff放在espnet根目录下，运行脚本生成encoder.onnx

      ```
      git checkout .
      git checkout v.0.10.5
      patch -p1 < export_onnx_dynamic.diff
      cd ./egs/aishell/asr1/
      bash export_onnx.sh
      ```


2. 开始推理验证。

   1. 获取精度

      ①静态shape

      首先修改acc.diff文件中的om模型路径（约162行）为生成的om路径

      ```
      cd espnet
      git checkout .
      git checkout v.0.10.5
      patch -p1 < acc.diff
      cd espnet/egs/aishell/asr1
      bash acc.sh
      ```

      ②动态shape

      首先修改acc_dynamic.diff文件中的om模型路径（约162行）为生成的om路径

      ```
      cd espnet
      git checkout .
      git checkout v.0.10.5
      patch -p1 < acc_dynamic.diff
      cd espnet/egs/aishell/asr1
      bash acc.sh
      ```

      即可打屏获取精度，精度保存在文件espnet/egs/aishell/asr1/exp/train_sp_pytorch_train_pytorch_conformer_kernel15_specaug/decode_test_decode_lm0.0_lm_4/result.txt

   2. 获取性能

      需要首先配置环境变量:

      ```
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      ```

      运行脚本infer_perf.py获取分档下的性能

      ```
      python3.7.5 infer_perf.py即可获取打印的fps性能
      ```

      运行脚本infer_perf_dynamic.py获取动态shape下的性能

      ```
      python3.7.5 infer_perf_dynamic.py即可获取打印的fps性能
      ```


# 模型推理性能精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用aclruntime推理计算，性能精度参考下列数据。

| 模型              | 310P性能   | T4性能     | 310P/T4 | 官网pth精度(Err)| 310P精度(Err) |
|-----------------|----------|----------|---------|---------|---------|
| Espnet_conformer | 分档：430fps；动态：25fps | 261fps | 1.647 | 5.1% | 5.4%|

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
