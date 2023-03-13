# CRNN_Dynamic-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)
  
- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

  ******




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

文字识别是图像领域一个常见问题。对于自然场景图像，首先要定位图像中的文字位置，然后才能进行文字的识别。对定位好的文字区域进行识别，主要解决的问题是每个文字是什么，将图像中的文字区域进转化为字符信息。CRNN全称为Convolutional Recurrent Neural Network，是一种卷积循环神经网络结构，用于解决基于图像的序列识别问题，特别是场景文字识别问题。主要用于端到端地对不定长的文本序列进行识别，不用先对单个文字进行切割，而是将文本识别转化为时序依赖的序列学习问题，也就是基于图像的序列识别。


- 参考实现：

  ```
  https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec
  branch=master
  commit_id=90c83db3f06d364c4abd115825868641b95f6181
  ```

  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                     | 数据排布格式 |
  | -------- | -------- | ------------------------ | ------------ |
  | input    | FLOAT32  | batchsize x 1 x 32 x len | NCHW         |


- 输出数据

  | 输出数据 | 大小                | 数据类型 | 数据排布格式 |
  | -------- | ------------------- | -------- | ------------ |
  | output1  | len x batchsize x 37 | FLOAT32  | ND           |



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



1. 安装依赖。

   ```
   pip3 install -r requirments.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持多种开源OCR mdb数据集（例如IIIT5K_lmdb），请用户自行准备好图片数据集，IIIT5K_lmdb验证集目录参考。

    ```
   ├── IIIT5K_lmdb        # 验证数据集
     ├── data.mdb         # 数据文件
     └── lock.mdb         # 锁文件
    ```

2. 数据预处理。

   1. 执行parse_testdata.py脚本。
   
   ```
   touch label.txt
   python3 parse_testdata.py --test_dir ./IIIT5K_lmdb --output_npy input_npy
   ```
   参数说明
   + test_dir：输入数据文件夹
   + output_npy: 结果保存文件夹
          
   
   执行成功后，结果保存在*./input_npy*文件夹下，生成标签数据label.txt。


## 模型推理<a name="section741711594517"></a>

模型转换。

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

获取权重文件。

因此模型官方实现没有提供对应的权重，所以此处使用NPU自行训练的权重结果作为原始输入，对应下载[权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/c-version/CRNN_for_PyTorch/zh/1.3/m/CRNN_for_PyTorch_1.3_model.zip)

下载后解压获取里面的checkpoint.pth权重文件

导出onnx文件。

1. 使用pth2onnx.py导出onnx文件。

   运行pth2onnx.py脚本。

   ```
   python3.7 pth2onnx.py ./checkpoint.pth ./crnn_dy_raw.onnx
   ```

   获得`crnn_dy_raw.onnx`文件。

2. 改图

   运行fix.py
   ```
   python3 fix.py crnn_dy_raw.onnx crnn_dy.onnx
   ```
   运行得到 `crnn_dy.onnx`

使用ATC工具将ONNX模型转OM模型。

3. 配置环境变量。

   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

   > **说明：** 
   > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

4. 执行命令查看芯片名称（$\{chip\_name\}）

      ```shell
      npu-smi info
      ```
      该设备芯片名为Ascend310P3 （请根据实际芯片填入）
      会显如下：

      ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

5. 执行atc命令

      ```shell
      # Ascend${chip_name}请根据实际查询结果填写 
      atc --model=crnn_dy.onnx --framework=5 --output=om/crnn_dy --input_format=ND --input_shape_range="imgs:[1~16,1,32,32~6400]" --log=error --soc_version=Ascend${chip_name}
      ```
      
      参数说明:  
      
      - --model：为ONNX模型文件 
      
      - --framework：5代表ONNX模型 
      
      - --output：输出的OM模型 
      
      - --input_format：输入数据的格式 
      
      - --input_shape_range：输入数据的shape范围
      
      - --log：日志级别 
      
      - --soc_version：处理器型号 
      
      运行成功后生成`om/crnn_dy.om`模型文件 

5. 开始推理验证。

    a. 安装ais_bench推理工具。

    请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

    b. 执行推理

    ```
    pytohn3 postpossess_CRNN_pytorch.py --input_npy input_npy --label ./label.txt --device 0 --om_path om/crnn_dy.om
    ```

    参数说明:

    - --input_npy: 输入数据 
    - --label: 标签地址 
    - --device：npu device id
    - --om_path: om 地址

    运行结果精度和npu E2E的耗时会打屏显示

6. GPU 性能推理

    > *特别说明：* 运行此脚本需要配套的GPU环境，保证cuda和TensorRT都正常安装。

    
    ```
    python3 gpu_end2end.py --input_npy input_npy --onnx_path crnn_dy_raw.onnx --engine_path crnn.trt
    ```
    - 参数说明
      - --input_npy: 输入数据
      - --onnx_path：onnx文件
      - --engine_path：trt文件

    运行结果gpu E2E耗时会打屏显示

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 |   数据集      | 精度   | E2E 耗时(s)  |
| -------- | ---------- | ----------- | ------ | 
| 310P3    | IIIT5K_lmdb | 66% | 5.9 ~ 6.1  |