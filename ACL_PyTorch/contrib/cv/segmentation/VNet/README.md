#  VNet 模型-推理指导

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

V-Net是一个早期的全卷积的三维图像分割网络，基本网络架构与2D图像分割网络U-Net相似，为了处理3D医学图像，采用了3D卷积模块和3D转置卷积模块。

- 参考论文：

  [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)

- 参考实现：

  ```
  url=https://github.com/mattmacy/vnet.pytorch
  branch=master
  commit_id=a00c8ea16bcaea2bddf73b2bf506796f70077687
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据      | 数据类型 | 大小                         | 数据排布格式 |
  | ------------- | -------- | ---------------------------- | ------------ |
  | actual_input1 | FP32     | batchsize x 1 x 64 x 80 x 80 | ND           |

- 输出数据

  | 输出数据 | 数据类型 | 大小                         | 数据排布格式 |
  | -------- | -------- | ---------------------------- | ------------ |
  | output1  | FP32     | batchsize x 64 x 80 x 80 x 2 | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1** 版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fpies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.12.1  | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取本仓源码。

2. 获取开源模型代码，和第1步源码置于同级目录下。

   ```
   git clone https://github.com/mattmacy/vnet.pytorch
   cd vnet.pytorch
   git checkout a00c8ea16bcaea2bddf73b2bf506796f70077687
   ```

3. 对源码进行修改，以满足数据集预处理及模型转换等功能。

   ```
   cd vnet.pytorch
   patch -p1 < ../vnet.patch
   cd ..
   ```

4. 安装必要依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   该模型使用[LUNA16数据集](https://luna16.grand-challenge.org/Download/)自行划分的测试集进行测试。这里提供已处理过的数据[下载链接](https://pan.baidu.com/s/1Vg8e6UISiWhpjsabSHCuew?pwd=55mc)，下载解压后上传至服务器，和第1步源码置于同级目录下，数据目录结构如下所示：

   ```
   ├── luna16  
         ├──normalized_lung_ct
              ├──1.3.6.1.4.1.14519.5.2.1.6279.6001.997611074084993415992563148335.mhd                    
              ├──...                     
         ├──normalized_lung_mask
              ├──1.3.6.1.4.1.14519.5.2.1.6279.6001.997611074084993415992563148335.mhd
              ├──...     
   ```
   
3. 执行预处理脚本，生成数据集预处理后的bin文件

   ```
   python3 vnet_preprocess.py ./luna16 ${prep_bin} ./test_uids.txt
   ```

   参数说明：

   - --参数1：数据集路径。
   - --参数2：指定保存bin文件的路径，比如 prep_bin。
   - --参数3：测试样本名文件路径。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      点击下载权重文件 [vnet_model_best.pth.tar](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/VNET/PTH/vnet_model_best.pth.tar)

   2. 导出onnx文件。

      1. 使用vnet_pth2onnx.py导出动态batch的onnx文件。

         ```
         python3 vnet_pth2onnx.py vnet_model_best.pth.tar vnet.onnx
         ```

         参数说明：

         - --参数1：模型权重文件路径。
         - --参数2：ONNX模型文件保存路径。

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
      # bs = [1, 4, 8, 16, 32, 64]
      atc --model=vnet.onnx --framework=5 --output=vnet_bs${bs} --input_format=NCDHW --input_shape="actual_input_1:${bs},1,64,80,80" --log=error --soc_version=Ascend${chip_name}
      ```

      运行成功后生成vnet_bs${bs}.om模型文件。

      参数说明：

      - --model：为ONNX模型文件。
      - --framework：5代表ONNX模型。
      - --output：输出的OM模型。
      - --input_format：输入数据的格式。
      - --input_shape：输入数据的shape。
      - --log：日志级别。
      - --soc_version：处理器型号。

2. 开始推理验证。

   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

   2. 执行推理。

      ```
      python3 -m ais_bench --model=vnet_bs${bs}.om  --batchsize=${bs} \
      --input ${prep_bin} --output result --output_dirname result_bs${bs} --outfmt BIN
      ```
      
      参数说明：
      
      - --model：om模型路径。
      - --batchsize：批次大小。
      - --input：输入数据所在路径。
      - --output：推理结果输出路径。
      - --output_dirname：推理结果输出子文件夹。
      - --outfmt：推理结果输出格式
   
3. 精度验证。

   调用vnet_postprocess.py脚本与真值标签比对，可以获得精度数据。

   ```
   python3 vnet_postprocess.py ${result_dir} ./luna16/normalized_lung_mask ./test_uids.txt
   ```

   参数说明：

   - --参数1：生成推理结果所在路径，比如这里为result/result_bs${bs}。
   - --参数2：真值标签数据路径。
   - --参数3：测试样本名文件路径。

4. 可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

   ```
   python3 -m ais_bench --model=vnet_bs${bs}.om --loop=50 --batchsize=${bs}
   ```

   参数说明：

   - --model：om模型路径。
   - --batchsize：批次大小。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，VNet模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集 | 开源精度（Point Accuracy） | 参考精度（Point Accuracy） |
| ----------- | ---------- | ------ | -------------------------- | -------------------------- |
| Ascend310P3 | 1          | LUNA16 | 99.645%                    | 99.409%                    |

| 芯片型号    | Batch Size | 参考性能（FPS） |
| ----------- | ---------- | --------------- |
| Ascend310P3 | 1          | 44.49           |
| Ascend310P3 | 4          | 43.12           |
| Ascend310P3 | 8          | 43.06           |
| Ascend310P3 | 16         | 42.88           |
| Ascend310P3 | 32         | 42.55           |
| Ascend310P3 | 64         | 41.94           |