# C3D模型-推理指导


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

C3D一种简单而有效的方法，用于使用在大规模监督视频数据集上训练的深层三维卷积网络（3D ConvNets）进行时空特征学习。该网络有三个方面的优势：1）与 2D ConvNets 相比，3D ConvNets 更适合于时空特征学习；2）所有层级的 3×3×3 小卷积核心 的均匀架构是 3D ConvNets 中性能最好的架构之一；3）使用简单的线性分类器学习的 特征，即 C3D（卷积 3D），在 4 个不同的基准上优于最先进的方法，并且与其他 2 个基准 上的当前最佳方法相当。另外，特征非常紧凑：仅使用 10 维的 UCF101 数据集的精度达到 52.8％，由于 ConvNets 的快速推理能力，其计算效率也非常高。最后，它们在概念上很简单，易于训练和使用。


- 参考实现：

  ```
  url=https://github.com/openmmlab/mmaction2/blob/master/configs/recognition/c3d
  branch=master
  commit_id=3e9e99ff7413b2b5c105586000dc0cc793ce00b5
  model_name=c3d
  ```
  


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                               | 数据排布格式 |
  | -------- | -------- | ---------------------------------- | ------------ |
  | image    | RGB_FP32 | batchsize x 10 x 3 x16 x 112 x 112 | NDCTHW       |


- 输出数据

  | 输出数据 | 数据类型 | 大小           | 数据排布格式 |
  | -------- | -------- | -------------- | ------------ |
  | class    | FP32     | batchsize x101 | ND           |




# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.8.0   | -                                                            |
| Torchvision                                                  | 0.9.1   |                                                              |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

1. 获取本仓源码。

2. 在同级目录下，获取第三方库mmaction2源码并安装。

   ```
   pip3 install openmim
   pip3 install mmcv-full==1.4.0
   
   git clone https://github.com/open-mmlab/mmaction2.git        # 克隆仓库的代码
   cd mmaction2                                                 # 切换到模型的代码仓目录
   git checkout 3e9e99ff7413b2b5c105586000dc0cc793ce00b5        # 切换到对应分支
   
   pip3 install -r requirements/build.txt
   pip3 install -v -e .
   ```

3. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   运行下述命令下载UCF101视频数据集并提取RGB原始帧（视频目录：mmaction2/data/ucf101/videos/，提取好的原始帧目录：mmaction2/data/ucf101/rawframes）

   ```
   cd tools/data/ucf101/
   bash download_videos.sh
   bash extract_rgb_frames_opencv.sh
   
   bash download_annotations.sh
   bash generate_rawframes_filelist.sh
   bash generate_videos_filelist.sh
   ```
   
   本仓代码和mmaction2源码的目录结构组织如下：
   
   ```
   C3D
   |-- C3D_postprocess.py
   |-- C3D_preprocess.py
   |-- C3D_pth2onnx.py
   |-- LICENSE
   |-- README.md
   |-- requirements.txt
   ├── mmaction2 #mmaction2的目录结构
       ├── mmaction
       ├── tools
       ├── configs
       ├── data
       │   ├── ucf101 #数据集目录结构
       │   │   ├── ucf101_{train,val}_split_{1,2,3}_rawframes.txt
       │   │   ├── ucf101_{train,val}_split_{1,2,3}_videos.txt
       │   │   ├── annotations
       │   │   ├── videos
       │   │   │   ├── ApplyEyeMakeup
       │   │   │   │   ├── v_ApplyEyeMakeup_g01_c01.avi  
       │   │   │   ├── YoYo
       │   │   │   │   ├── v_YoYo_g25_c05.avi
       │   │   ├── rawframes
       │   │   │   ├── ApplyEyeMakeup
       │   │   │   │   ├── v_ApplyEyeMakeup_g01_c01
       │   │   │   │   │   ├── img_00001.jpg
       │   │   │   │   │   ├── img_00002.jpg
       │   │   │   │   │   ├── ...
       │   │   │   │   │   ├── flow_x_00001.jpg
       │   │   │   │   │   ├── flow_x_00002.jpg
       │   │   │   │   │   ├── ...
       │   │   │   │   │   ├── flow_y_00001.jpg
       │   │   │   │   │   ├── flow_y_00002.jpg
       │   │   │   ├── ...
       │   │   │   ├── YoYo
       │   │   │   │   ├── v_YoYo_g01_c01
       │   │   │   │   ├── ...
       │   │   │   │   ├── v_YoYo_g25_c05
   ```
   
2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行C3D_preprocess.py脚本，将原始帧（rawframes）处理为bin文件。

   ```
   cd ${path_to_C3D}/mmaction2
   mkdir prep_datasets
   python3 ../C3D_preprocess.py ./configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py --output_path ./prep_datasets
   ```
   
	参数说明：
	
	- --参数1：配置文件的路径.
	- --output_path：输出文件夹的位置


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       [pth文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/C3D/PTH/C3D.pth)

   2. 导出onnx文件。

      1. 使用C3D_pth2onnx.py导出onnx文件。

         确保当前目录为${path_to_C3D}/mmaction2，运行C3D_pth2onnx.py脚本，获得动态batch的C3D.onnx文件。

         ```
         python3 ../C3D_pth2onnx.py ./configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py ../C3D.pth \
         --output-file C3D.onnx --shape 1 10 3 16 112 112  --verify --softmax
         ```
         
         参数说明：
         
         - --参数1：模型配置文件。
         - --参数2：模型权重文件。
         - --output-file：导出的onnx文件。
         - --shape: 模型输入张量的形状。对于C3D模型，输入形状为 `${batch} x ${clip} x ${channel} x ${time} x ${height} x ${width}`。
         - --verify: 决定是否对导出模型进行验证，验证项包括是否可运行，数值是否正确等。如果没有被指定，它将被置为 False。
         - --softmax: 是否在行为识别器末尾添加 Softmax。如果没有指定，将被置为 False。目前仅支持行为识别器，不支持时序动作检测器。
         
         > **说明：** 
         > 该步骤与依赖版本相关，建议参照requirements.txt安装对应版本的torch和torchvision。
      
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
         # bs=[1,4,8,16,32]
         atc --framework=5 --model=C3D.onnx --output=C3D_bs${bs} --input_format=ND --input_shape="image:${bs},10,3,16,112,112" --log=error --soc_version=Ascend${chip_name}
         ```
         
         参数说明：
         
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
         
           运行成功后生成C3D.om模型文件。

2. 开始推理验证。
    a. 安装ais_bench推理工具

       请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。 
	
    b.  执行推理。
	```shell
	# 移除异常数据
    bash ../check_rawframes_filelist.sh
    rm -rf prep_datasets/v_PommelHorse_g05*.bin
    
    # 执行推理
    python3 -m ais_bench --model ./C3D_bs${bs}.om --batchsize=${bs} --input=./prep_datasets/ --output ./result --output_dirname result_bs${bs} --outfmt TXT
    ```
    
    参数说明：
    
    - --model：需要进行推理的om模型。
    - --input：模型需要的输入，支持bin文件和目录，若不加该参数，会自动生成都为0的数据。
    - --output：推理结果输出路径。默认会建立日期+时间的子文件夹保存输出结果 如果指定output_dirname 将保存到output_dirname的子文件夹下。
    - --outfmt：输出数据的格式，默认”BIN“。
    - --batchsize：模型batch size 默认为1 。当前推理模块根据模型输入和文件输出自动进行组batch。参数传递的batchszie有且只用于结果吞吐率计算。请务必注意需要传入该值，以获取计算正确的吞吐率。
    - --output_dirname：推理结果输出子文件夹。可选参数。与参数output搭配使用，单独使用无效。设置该值时输出结果将保存到 output/output_dirname文件夹中。
    
    
    推理后的输出在当前目录result/result_bs${bs}下。
    
    c.  精度验证。
    
    ```shell
    python3 ../C3D_postprocess.py ./result/result_bs${bs}/ ./data/ucf101/ucf101_val_split_1_rawframes.txt ./top1_acc.json
    ```
    
    参数说明：
    - --参数1：离线推理得到的结果文件夹所在的路径
    - --参数2：标注文件所在的路径
    - --参数3：输出的json文件保存路径，json文件中保存了精度数据
    
    运行之后会在当前目录生成top1_acc.json文件，保存精度数据。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，C3D的精度和性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度指标（Top-1） | 性能（FPS） |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| 310P | 1 | UCF101 | 81.87% | 51.61 |
| 310P | 4 | UCF101 | 81.87% | 54.92 |
| 310P | 8 | UCF101 | 81.87% | 52.65 |
| 310P | 16 | UCF101 | 81.87% | 52.24 |
| 310P | 32         | UCF101 | 81.87% | 50.72 |