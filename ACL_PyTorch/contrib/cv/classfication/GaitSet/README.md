# {GaitSet}模型-推理指导


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

GaitSet是一个灵活、有效和快速的跨视角步态识别网络，迁移自https://github.com/AbnerHqC/GaitSet



- 参考实现：

  ```
  url=https://github.com/AbnerHqC/GaitSet
  branch=master
  commit_id=14ee4e67e39373cbb9c631d08afceaf3a23b72ce
  model_name=GaitSet
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

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 100 x 64 x 44 | NCHW         |


- 输出数据

  | 输出数据 | 大小         | 数据类型 | 数据排布格式 |
  | -------- | ------------ | -------- | ------------ |
  | output1  | 1 x 62 x 256 | FLOAT32  | ND           |




# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本     | 环境准备指导                                                 |
| ------------------------------------------------------------ | -------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15   | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2  | -                                                            |
| Python                                                       | 3.7.5    | -                                                            |
| PyTorch                                                      | 1.5.0    | -                                                            |
| onnx                                                         | 1.7.0    | -                                                            |
| opencv-python                                                | 4.5.2.52 | -                                                            |
| numpy                                                        | 1.20.1   | -                                                            |
| imageio                                                      | 2.9.0    | -                                                            |
| xarray                                                       | 0.18.2   | -                                                            |
| sympy                                                        | 1.10.1   | -                                                            |
| six                                                          | 1.16.0   | -                                                            |
| wheel                                                        | 0.37.1   | -                                                            |
| decorator                                                    | 5.1.1    | -                                                            |
| mpmath                                                       | 1.2.1    | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \        | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>



1. 安装依赖。

   ```
   pip3 install -r requirment.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持CASIA-B图片的验证集。下载地址http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp  ，只下载DatasetB数据集。

   下载后的数据集内的压缩文件需要全部解压，解压后数据集内部的目录应为（`CASIA-B`数据集）：数据集路径/对象序号/行走状态/角度，例如`CASIA-B/001/nm-01/000/ `。

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   a.执行命令编辑脚本。

   ```
   vim config_1p.py 
   #修改dataset_path为b命令pretreatment.py中output_path所用的路径
   执行:wq保存退出编辑。
   ```

   b.执行命令，完成数据集预处理。

   ```
   python pretreatment.py --input_path='root_path_of_raw_dataset' --output_path='root_path_for_output'
   ```

   第一个参数是数据集所在目录，第二个参数是预处理后的文件名

   c.执行命令生成bin文件夹。

   ```
   mkdir CASIA-B-bin
   python -u test.py --iter=-1 --batch_size 1 --cache=True --pre_process=True
   ```

   d.执行命令生成info文件。

   ```
   python gen_dataset_info.py bin CASIA-B-bin CASIA-B-bin.info 64 64
   ```

**说明：** 

预处理过程中提示大量`WARNING`属于正常现象。如果出现`ERROR`错误提示则可能路径设置有误、或要求中的库文件没有安装。由于`ERROR`提示等重新导出时，建议删除导出有误的文件后再导出。

运行时，首先初步处理后的数据集会在导出路径下生成。

随后，脚本会使用生成的数据集，在当前根目录下生成`CASIA-B-bin`文件夹，里面含有处理好的二进制格式的图片。之后，脚本会在当前根目录下生成以`.info`结尾的图片列表文件，用于推理。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       在源码包中已经提供权重文件GaitSet_CASIA-B_73_False_256_0.2_128_full_30-80000-encoder.ptm，如果没有，可以使用源码自带的ptm进行推理，地址：https://github.com/AbnerHqC/GaitSet/tree/master/work/checkpoint/GaitSet 。进入此地址下载里面的encoder.ptm后缀的文件

   2. 导出onnx文件，此处导出的onnx为静态，因此需要每个batch_size的onnx。

      1. 代码转换为静态的onnx，需在代码中修改batchsize大小。

         a.执行命令编辑脚本。

         ```
         vim pth2onnx.py 
         #修改dummy_input = torch.randn((1, align_size, 64, 44)) 中第一个参数为需要的batchsize
         执行:wq保存退出编辑。
         ```

      2. 使用pth2onnx.py导出onnx文件。

         运行pth2onnx.py脚本，获得gaitset_submit.onnx文件。

         ```
         python pth2onnx.py –-input_path=’${权重文件路径}’
         ```
      
   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         bash env.sh
         ```
   
         > **说明：** 
   >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
   
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
         atc --framework=5 --model=gaitset_submit.onnx --output=gaitset_submit --input_shape="image_seq:1,100,64,44" --log=debug --soc_version=Ascend310P3
         ```
   
         - 参数说明：
   
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --insert\_op\_conf=aipp\_resnet34.config:  AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。
   
   
   
   
      运行成功后生成gaitset_submit.om模型文件。



2. 开始推理验证。

   a.  使用ais-infer工具进行推理。

   执行命令增加工具可执行权限，并根据OS架构选择工具

   ```
   chmod u+x 
   ```

   b.  执行推理。

    纯推理模式：
    ```
    python ais_infer.py --model /home/trc/GaitSet/gaitset_submit_bs1_310P.om --batchsize 8 --loop 10
    ```
    
    -   参数说明：
    
        -   batchsize：batchsize大小。
        -   loop：推理次数，可选参数，默认1，profiler为true时，推荐为1。
    	...
    	
    真实数据推理：
    ```
     python ais_infer.py --model gaitset_submit_bs1_310P3.om --input "CASIA-B-bin"
    ```
    
    -   参数说明：
    
        -   model：om文件路径。
        -   input：输入数据。
    	

 

   c.  精度验证。

    执行`eval_acc_perf.sh`：
    
    ```bash
    bash test/eval_acc_perf.sh
    ```
    
    或者在配置好了环境的前提下直接运行：
    
    ```bash
    python -u test.py --iter=-1 --batch_size 1 --cache=True --post_process=True
    ```
    
    参数`--iter`、`--cache`、`--post_process`为模型后处理固定参数不需修改。



   原模型精度95.405%：



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| 310      | 1          | CASIA-B  DatasetB | Rank1:95.512% | 599.98  |
| 310 | 4 | CASIA-B  DatasetB | | 667.168 |
| 310 | 8 | CASIA-B  DatasetB | | 678.32 |
| 310 | 16 | CASIA-B  DatasetB | Rank1:95.512% | 684.812 |
| 310 | 32         | CASIA-B  DatasetB | | 681.212 |
| 310 | 64 | CASIA-B  DatasetB | | 681.564 |
| 310P | 1 | CASIA-B  DatasetB | Rank1:95.512% | 849.55 |
| 310P | 4 | CASIA-B  DatasetB | | 907.832 |
| 310P | 8 | CASIA-B  DatasetB |  | 926.033 |
| 310P | 16 | CASIA-B  DatasetB | Rank1:95.512% | 941.825 |
| 310P | 32 | CASIA-B  DatasetB |  | 950.833 |
| 310P | 64 | CASIA-B  DatasetB |  | 952.93 |
| T4 | 1 | CASIA-B  DatasetB |  | 354.39 |
| T4 | 4 | CASIA-B  DatasetB |  | 395.37 |
| T4 | 8 | CASIA-B  DatasetB |  | 388.48 |
| T4 | 16 | CASIA-B  DatasetB |  | 379.23 |
| T4 | 32 | CASIA-B  DatasetB |  | 394.04 |
| T4 | 64 | CASIA-B  DatasetB |  | 384.21 |



以上在310P上的结果为AOE优化后的性能。