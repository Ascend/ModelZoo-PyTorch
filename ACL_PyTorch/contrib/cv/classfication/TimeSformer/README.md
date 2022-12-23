# TimeSformer模型-推理指导


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

TimeSformer论文提出了一种无卷积的视频分类方法，该方法专门基于名为“TimeSformer”的空间和时间上的自注意力而构建，通过直接从一系列帧级块中启用时空特征学习，将标准的Transformer体系结构应用于视频，使得它能够捕捉到整段视频中的时空依赖性。


- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/timesformer
  branch=master
  commit_id=f58e10d0d6789f135f8ae5028cc982a299d8badd
  model_name=timesformer_divST_8x32x1_15e_kinetics400_rgb
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | imgs    | float32 | 1 x 3 x 3 x 8 x 224 x 224 | ND         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | result  | 3 x 400 | float32  | ND           |



# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.7.1   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/open-mmlab/mmaction2.git
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   cd mmaction2
   pip3 install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.7.0/index.html
   pip3 install -r requirements/build.txt
   pip3 install -v -e .
   cd ..
   ```

3. 应用补丁，修改模型代码。

   ```
   patch -p1 < TimeSformer.patch
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   下载[kinetics400数据集](https://github.com/cvdfoundation/kinetics-dataset)

   软链接下载好的数据集到 mmaction2 文件夹中的相应位置处
   ```
   mkdir -p mmaction2/data/kinetics400/annotations/

   ln -s ${data_path}/kinetics-dataset/k400/annotations/test.csv mmaction2/data/kinetics400/annotations/kinetics_test.csv

   ln -s ${data_path}/kinetics-dataset/k400/annotations/val.csv mmaction2/data/kinetics400/annotations/kinetics_val.csv

   ln -s ${data_path}/kinetics-dataset/k400/annotations/train.csv mmaction2/data/kinetics400/annotations/kinetics_train.csv

   ln -s ${data_path}/kinetics-dataset/k400/val/ mmaction2/val
   ```
   进入mmaction2文件夹，将数据集视频做抽帧处理
   ```
   cd mmaction2

   python3 tools/data/build_rawframes.py val data/kinetics400/rawframes_val/ --level 1 --ext mp4 --task rgb --new-short 320 --use-opencv

   python3 tools/data/build_file_list.py kinetics400 data/kinetics400/rawframes_val/ --level 1 --format rawframes --num-split 1 --subset val --shuffle

   cd ..
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行TimeSformer_preprocess.py脚本，完成预处理。

   ```
   python3 TimeSformer_preprocess.py
   ```
   预处理后的bin文件在./out_bin，info文件为 `k400.info`

   参数说明：  
   ${data_path}/kinetics-dataset/k400/val：验证集路径  
   ./out_bin：预处理后的 bin 文件存放路径，每个图像对应生成一个二进制文件。



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      将权重文件下载到本地后上传到当前工作目录。
      ```
      wget https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb/timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth
      ```

   2. 导出onnx文件。

      1. 使用模型代码仓pth2onnx导出onnx文件。

         运行pth2onnx脚本。

         ```
         python3 mmaction2/tools/deployment/pytorch2onnx.py mmaction2/configs/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb.py timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth --shape 1 3 3 8 224 224 --verify --output-file tsf.onnx
         ```

         获得tsf.onnx文件。

      2. 优化ONNX文件。

         ```
         python3 -m onnxsim --input-shape="1,3,3,8,224,224" tsf.onnx tsf_sim.onnx
         ```

         获得tsf_sim.onnx文件。

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
         atc --framework=5 \
         --model=tsf_sim.onnx \
         --output=tsf \
         --input_format=ND \
         --input_shape="imgs:1,3,3,8,224,224" \
         --log=error \
         --soc_version=Ascend${chip_name} \
         --op_select_implmode=high_performance \
         --optypelist_for_implmode="Gelu"
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --op_select_implmode：设置网络模型中所有算子为高性能实现。
           -   --optypelist_for_implmode：列举算子optype的列表。

           运行成功后生成**tsf.om**模型文件。



2. 开始推理验证。

   a.  安装ais_bench推理工具。

      请点击本链接进行安装ais_bench推理工具，以及查看具体使用方法(https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)  


   b.  执行推理。

      ```
      # 创建 result 文件夹，存放推理结果文件
      mkdir result

      # 推理
      python3 -m ais_bench  --model ./tsf.om --input ./out_bin --output ./result --outfmt TXT
      ```

      -   参数说明：

           -  --model：om文件路径。
           -  --input：输入名及文件路径。
           -  --output：输出路径。
           -  --outfmt：输出文件格式。

      推理后的输出默认在当前目录result下。

      >**说明：** 
      >执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见[《ais_bench 推理工具使用文档》](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)。

   c.  精度验证。

      调用脚本与数据集标签k400.info比对。

      ```
       python3 TimeSformer_postprocess.py --result_path ./result/2022_xx_xx-xx_xx_xx/ --info_path=k400.info
      ```

      -  参数说明：
         -  `./result/2022_xx_xx-xx_xx_xx/` 中的 2022_xx_xx-xx_xx_xx 为 ais_bench 自动生成的目录名。  
         -  `k400.info`为数据预处理时生成的info文件


   d.  性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench \
      --model=./tsf.om \
      --loop 50
      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|Ascend 310P3|         1   |   Kinetics400  |top1 acc= 77.68 |    7.53 fps   |

注：模型只支持bs1