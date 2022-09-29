# CRNN-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

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
  | input    | FLOAT32  | batchsize x 1 x 32 x 100 | NCHW         |


- 输出数据

  | 输出数据 | 大小                | 数据类型 | 数据排布格式 |
  | -------- | ------------------- | -------- | ------------ |
  | output1  | 26 x batchsize x 37 | FLOAT32  | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.6.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>



1. 安装依赖。

   ```
   pip3 install -r requirment.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持多种开源OCR mdb数据集（例如IIIT5K_lmdb），请用户自行准备好图片数据集，IIIT5K_lmdb验证集目录参考。

    ```
   ├── IIIT5K_lmdb        # 验证数据集
     ├── data.mdb         # 数据文件
     └── lock.mdb         # 锁文件
    ```

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   修改脚本参数。          

   执行命令。           

   ```
   vim parse_testdata.py
   ```

   修改脚本中test_dir和output_bin参数值。           

   ```
   test_dir = '/home/HwHiAiUser/dataset/IIIT5K_lmdb' # 修改为实际数据实际存放路径
   output_bin = './input_bin/'     # 修改为二进制输出路径
   ```

   执行parse_testdata.py脚本。

   ```
   python3 parse_testdata.py
   ```

   执行成功后，二进制文件生成在*./input_bin*文件夹下，标签数据label.txt生成在当前目录下。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      因此模型官方实现没有提供对应的权重，所以此处使用NPU自行训练的权重结果作为原始输入，对应下载[权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/c-version/CRNN_for_PyTorch/zh/1.3/m/CRNN_for_PyTorch_1.3_model.zip)

      下载后解压获取里面的checkpoint.pth权重文件

   2. 导出onnx文件。

      1. 使用pth2onnx.py导出onnx文件。

         运行pth2onnx.py脚本。

         ```
         python3.7 pth2onnx.py ./checkpoint.pth ./crnn_npu_dy.onnx
         ```

         获得crnn_npu_dy.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （请根据实际芯片填入）
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
         atc --model=crnn_npu_dy.onnx --framework=5 --output=crnn_final_bs16 --input_format=NCHW --input_shape="actual_input_1:16,1,32,100" --log=info --soc_version=Ascend${chip_name}
         
         备注：Ascend${chip_name}请根据实际查询结果填写
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --insert\_op\_conf:  AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。

           运行成功后生成crnn_final_bs16.om模型文件。

           

2.开始推理验证。

a.  使用ais-infer工具进行推理。

参考[ais-infer工具源码地址](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)安装将工具编译后的压缩包放置在当前目录；解压工具包，安装工具压缩包中的whl文件；

```
 pip3 install aclruntime-0.01-cp37-cp37m-linux_xxx.whl
```

b.  执行推理。

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    python3 ais_infer.py --model ./crnn_sim_16bs.om --input ./input_bin --output ./ --output_dirname result --device 0 --batchsize 16 --output_batchsize_axis 1
        
    -   参数说明：   
        --model：模型地址
        --input：预处理完的数据集文件夹
        --output：推理结果保存地址
        --outfmt：推理结果保存格式
    
    运行成功后会在result/xxxx_xx_xx-xx-xx-xx（时间戳）下生成推理输出的txt文件。
    
    **说明：** 
    执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见 --help命令。

**因工具限制，需要把result/xxxx_xx_xx-xx-xx-xx/summary.json从结果目录中删除，或者迁移到其他目录；**

c.  精度验证。

运行脚本postpossess_CRNN_pytorch.py进行精度测试，精度会打屏显示。

    python3 postpossess_CRNN_pytorch.py

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集      | 精度   | 性能 |
| -------- | ---------- | ----------- | ------ | ---- |
| 310P3    | 16         | IIIT5K_lmdb | 79.27% | 4400 |
