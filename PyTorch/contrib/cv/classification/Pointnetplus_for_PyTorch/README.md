# Pointnet++模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Pointnetplus是针对3D点云进行数据分类与分割的网络，该网络借鉴cnn的多层感受野的思想，通过不断的扫描增加感受野提取局部特征，相比Pointnet网络，一是通过多次set abstraction进行多层次提取特征来改善Pointnet在复杂场景中的应用，二是提出了多尺度分组和多分辨率分组改进非均匀点云的处理。


- 参考实现：

```
url=https://github.com/yanx27/Pointnet_Pointnet2_pytorch
branch=master
commit_id=e365b9f7b9c3d7d6444278d92e298e3f078794e1
```

  通过Git获取对应commit\_id的代码方法如下：

```
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch models
cd models
git checkout e365b9f7b9c3d7d6444278d92e298e3f078794e1
patch -p1 < ../models.patch
cd ..
```
- 目录结构：

```
    ├── pointnetplus_postprocess.py   //验证推理结果，给出Accuracy
    ├── pointnetplus_pth2onnx.py      //用于转换pth文件到onnx文件
    ├── pointnetplus_preprocess.py    /数据集预处理脚本
    ├── README.md 
    ├── models.patch                   //模型patch
    ├── requirements.txt              //运行环境
```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input1    | FP32 | batchsize x 3 x 32 x 512 | ND         |
  | input1    | FP32 | batchsize x 131 x 64 x 128 | ND         |
  | input2   | FP32 | batchsize x 3 x 128 | ND         |

- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | batchsize x 128 x 512 | FP32  | ND           |
  | output1  | batchsize x 40 | FP32  | ND           |
  | output2  | batchsize x 1024 x 1 | FP32  | ND           |


# 推理环境准备

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

| 配套 | 版本 | 环境准备指导 |
| ------- | ------- | ------- |
| 固件与驱动  | 22.0.2 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN  | 5.1.RC2 | - |
| Pytorch | 1.9.0 | - |

​       **表 2**  环境依赖表

| 配套 | 版本 |
| ------- | ------- |
| onnx  | 1.9.0|
| torch  | 1.9.0|
| torchvision | 0.10.0|
| numpy | 1.20.2|
| tqdm | 4.62.2|

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>


1. 安装依赖。

    ```
    pip install -r requirements.txt
    ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   Modelnet40数据集，上传数据集到服务器任意目录并解压（以/opt/npu/modelnet40_normal_resampled为例）

    ```
    ├── data
        ├── modelnet40_normal_resampled 
    ```

   modelnet40_normal_resampled.zip数据集下载链接： https://blog.csdn.net/Shertine/article/details/124091578

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。需要对模型的两部分都进行预处理，其中part1部分，需要输入原始数据；part2部分需要输入part1生成的数据(xyz_chg_part1)	以及om模型推理的结果（./Pointnetplus/out/bs1/part1）。

   执行pointnetplus_preprocess.py脚本，完成预处理。

   说明：详细部分可见本文档ais_bench推理的部分。由于需要om模型生成的中间结果，因此，此步骤可以先不执行，到推理部分再执行。

    ```
    python pointnetplus_preprocess.py --preprocess_part 1 --save_path ./modelnet40_processed/part1/pointset_chg_part1 --save_path2 ./modelnet40_processed/part1/xyz_chg_part1 --data_loc $datasets_path

    python pointnetplus_preprocess.py --preprocess_part 2 --save_path ./Pointnetplus/modelnet40_processed/bs1/pointset_chg_part2 --save_path2 ./Pointnetplus/modelnet40_processed/bs1/xyz_chg_part2 --data_loc ./Pointnetplus/out/bs1/part1 --data_loc2 ./Pointnetplus/modelnet40_processed/part1/xyz_chg_part1
    ```

   - 参数说明：
       -   --preprocess_part: 当前处理的是第几部分的文件。
       -   --save_path：输出数据保存路径1，主要保存两部分生成的pointset_chg_part*。
       -   --save_path2：输出数据保存路径2，主要保存两部分生成的xyz_chg_part*。
       -   --data_loc：数据集路径, 第一部分中输入原始数据集的路径，第二部分中输入om模型对第一部分进行推理后生成的数据路径（./Pointnetplus/out/bs1/part1）。
       -   --data_loc2: 数据集路径, 对原始数据进行处理生成的xyz_chg_part1文件所在的路径。

      每个图像对应生成一个二进制文件。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       从源码包中获取权重文件：“best_model.pth”，解压checkpoints.zip。

       源码包获取链接：

       ```
       git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch models
       ```

       比如best_model的path：Pointnetplus/models/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth

   2. 导出onnx文件。

      1. 使用best_model.pth导出onnx文件。

         运行pointnetplus_pth2onnx.py脚本。

         ```
         python pointnetplus_pth2onnx.py --target_model 1 --pth_dir './models/log/classification/pointnet2_ssg_wo_normals/checkpoints' --batch_size 1

         python pointnetplus_pth2onnx.py --target_model 2 --pth_dir './models/log/classification/pointnet2_ssg_wo_normals/checkpoints' --batch_size 1
         ```

         获得“Pointnetplus_part1_bs1.onnx”和“Pointnetplus_part2_bs1.onnx”文件，分别导出part1和part2。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（${chip_name}）。

         ```
         npu-smi info
         # 该设备芯片名为Ascend310P3 （自行替换）
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
         atc --framework=5 --model=Pointnetplus_part1_bs1.onnx --output=Pointnetplus_part1_bs1 --input_format=ND --input_shape="samp_points:1,3,32,512" --log=debug --soc_version=${chip_name}

         atc --framework=5 --model=Pointnetplus_part2_bs1.onnx --output=Pointnetplus_part2_bs1 --input_format=ND --input_shape="l1_points:1,131,64,128;l1_xyz:1,3,128" --log=debug --soc_version=${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件（以bs1的Pointnetplus_part1_bs1.onnx为例）。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成Pointnetplus_part1_bs1.om和Pointnetplus_part2_bs1.om模型文件。



2. 开始推理验证。

a.  使用ais_bench工具进行推理。

   执行命令增加工具可执行权限，并根据OS架构选择工具

   [ais_bench推理工具的安装方法](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)

   ```
从tools中将ais_infer文件夹复制到与om模型同级的目录下
cd ais_infer
   ```

b.  执行推理。

推理流程： 
1. 先对数据进行预处理，获得part1数据（pointset_chg_part1和xyz_chg_part1）
```
 python pointnetplus_preprocess.py --preprocess_part 1 --save_path /home/houyw/Pointnetplus/modelnet40_processed/part1/pointset_chg_part1 --save_path2 /home/houyw/Pointnetplus/modelnet40_processed/part1/xyz_chg_part1 --data_loc /home/houyw/modelnet40_normal_resampled
```
2. 使用ais_bench对part1进行推理，并对结果进行保存（./out/bs1/part1）
3. 对part1的数据进行预处理，获得part2数据（输入./out/bs1/part1和xyz_chg_part1, 输出pointset_chg_part2和xyz_chg_part2）
4. 使用ais_bench对part2进行推理，并对结果进行保存（./out/bs1/part2）

```
   mkdir -p /home/houyw/Pointnetplus/out/bs1/part1

   cd ais_infer

   python -m ais_bench  --model ../Pointnetplus_part1_bs1.om --input '/home/houyw/Pointnetplus/modelnet40_processed/part1/xyz_chg_part1,/home/houyw/Pointnetplus/modelnet40_processed/part1/pointset_chg_part1' --output '/home/houyw/Pointnetplus/out/bs1/part1' --outfmt BIN --loop 10 --batchsize 1

   cd ..

   python pointnetplus_preprocess.py --preprocess_part 2 --save_path /home/houyw/Pointnetplus/modelnet40_processed/bs1/pointset_chg_part2 --save_path2 /home/houyw/Pointnetplus/modelnet40_processed/bs1/xyz_chg_part2 --data_loc /home/houyw/Pointnetplus/out/bs1/part1 --data_loc2 /home/houyw/Pointnetplus/modelnet40_processed/part1/xyz_chg_part1

   mkdir -p /home/houyw/Pointnetplus/out/bs1/part2

   cd ais_infer

   python -m ais_bench  --model ../Pointnetplus_part2_bs1.om --input '/home/houyw/Pointnetplus/modelnet40_processed/part1/xyz_chg_part2,/home/houyw/Pointnetplus/modelnet40_processed/part1/pointset_chg_part2' --output '/home/houyw/Pointnetplus/out/bs1/part2' --outfmt BIN --loop 10 --batchsize 1
```

- 参数说明：

    - --model：需要进行推理的om模型。

    - --output：推理数据输出路径。

    - --outfmt：输出数据的格式，默认”BIN“，可取值“NPY”、“BIN”、“TXT”。

    - --loop: 推理次数，可选参数，默认1，profiler为true时，推荐为1。

    - --batchsize: 模型batch size 默认为1 。当前推理模块根据模型输入和文件输出自动进行组batch。参数传递的batchszie有且只用于结果吞吐率计算。请务必注意需要传入该值，以获取计算正确的吞吐率。

  推理后的输出默认在当前目录result下。

c.  精度验证。

执行pointnetplus_postprocess.py脚本直接可以生成精度结果。 输入part2推理路径以及数据集所在路径。

```python
    python pointnetplus_postprocess.py --target_path $output_dir --data_loc $datasets_path

    # 示例命令：
    python pointnetplus_postprocess.py --target_path /home/houyw/Pointnetplus/out/bs1/part2 --data_loc /home/houyw/modelnet40_normal_resampled
```
- 参数说明：
    - --target_path：生成推理结果所在路径。
    - --data_loc：数据集路径。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

精度：
|  | Instance Acc  | Class Acc |
|  :--:  | :--:  |  :--:  |
| 官网 | 0.928964 | 0.890532 |
| 310 | 0.926256 | 0.885494 |
| 310P | 0.928687 | 0.889116 |

性能：
|  Throughput   | 310(benchmark) | 310P(ais_infer)  |310P(ais_infer)/310 |
|  :--:  | :--:  |  :--:  | :--:  |
| PointNet++ bs1 part1 | 5747.24 | 7338 |  1.276786771 |
| PointNet++ bs1 part2 | 3223.88 | 4705.88 |  1.459694529 |
| PointNet++ bs4 part1 | 5694.56 | 6527.41 |  1.146253617 |
| PointNet++ bs4 part2 | 3527.608 | 4345.46 |  1.231843220 |
| PointNet++ bs8 part1 | 5990.2 | 6082.73 |  1.015446896 |
| PointNet++ bs8 part2 | 3751.416 | 3807.17 | 1.014862121 |
| PointNet++ bs16 part1 | 5856.8 | 5891.02 | 1.005842781 |
| PointNet++ bs16 part2 | 3807.368 | 3905.20 | 1.025695441 |
| PointNet++ bs32 part1 | 5946 | 6100.93 | 1.026056172 |
| PointNet++ bs32 part2 | 3912.636 | 4011.94 |  1.025380331 |
| PointNet++ bs64 part1 | 5766.428 | 6236 |  1.081432204 |
| PointNet++ bs64 part2 | 3711.3659 | 4093 |  1.10282848 |
| 最优batch part1 | 5990.2 | 7338 | 1.225000835 |
| 最优batch part2 | 3912.636 | 4705.88 | 1.202739023 |
