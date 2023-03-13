# SOLOV1模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

   - [输入输出数据](#ZH-CN_TOPIC_0000001126281702)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

SOLOV1模型是一个box-free的实例分割模型，其引入“实例类别”的概念来区分图像中的对象实例，即量化的中心位置和对象大小，这使得可以利用位置来分割对象。与其他端到端的实例分割模型相比，其达到了竞争性的准确性。

- 参考实现：

  ```
  url=https://github.com/WXinlong/SOLO
  branch=master
  commit_id= 95f3732d5fbb0d7c7044c7dd074f439d48a72ce5
  model_name=SOLOV1
  code_path=SOLO/tree/master/configs/solo
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                       | 数据排布格式 |
  | -------- | -------- | -------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 800 x 1216 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小        | 数据排布格式 |
  | -------- | -------- | ----------- | ------------ |
  | output1  | FLOAT32  | 100x200x304 | ND           |
  | output2  | INT32    | 100         | ND           |
  | output3  | FLOAT32  | 100         | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                            | 版本    | 环境准备指导                                                                                          |
  | --------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
  | 固件与驱动                                                      | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 5.1.RC2 | -                                                                                                     |
  | Python                                                          | 3.7.13  | -                                                                                                     |
  | PyTorch                                                         | 1.9.0   | -                                                                                                     |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |

- 该模型需要以下依赖

  **表 1**  依赖列表

  | 依赖名称        | 版本     |
  | --------------- | -------- |
  | torch           | 1.9.0    |
  | torchvision     | 0.10.0   |
  | onnx            | 1.9.0    |
  | onnx-simplifier | 0.3.6    |
  | onnxruntime     | 1.8.0    |
  | numpy           | 1.21.0   |
  | Cython          | 0.29.25  |
  | Opencv-python   | 4.5.4.60 |
  | pycocotools     | 2.0.3    |
  | Pytest-runner   | 5.3.1    |
  | protobuf        | 3.20.0   |
  | decorator       | \        |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码

1. 获取SOLOv1源代码。

   在已下载的源码包根目录下，执行如下命令。

   ```
   git clone https://github.com/WXinlong/SOLO.git -b master
   cd SOLO
   git reset --hard 95f3732d5fbb0d7c7044c7dd074f439d48a72ce5
   patch -p1 < ../MMDET.diff
   patch -p1 < ../SOLOV1.diff
   pip install -r requirements/build.txt
   pip install -v -e .
   cd ..
   ```

2. 安装依赖。

   ```
   pip install -r requirements.txt
   ```

3. 编译安装mmcv。

   ```
   git clone https://github.com/open-mmlab/mmcv -b v0.2.16
   cd mmcv
   python setup.py build_ext
   python setup.py develop
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型需要coco2017数据集，数据集下载[地址](https://cocodataset.org/)

   请根据实际需求将val2017图片及其标注文件拷贝到服务器指定目录，这里以 `dataset=/root/dataset/coco/` 为例进行后续操作。

   val2017目录存放coco数据集的验证集图片，annotations目录存放coco数据集的instances_val2017.json，文件目录结构如下：
   ```
    root
    ├── dataset
    │   ├── coco
    │   │   ├── annotations
    │   │   ├── val2017
   ```

2. 数据预处理。

   将原始数据集转换为模型输入的二进制数据。执行“solov1_preprocess.py”脚本。
   ```shell
   python solov1_preprocess.py \
         --image_src_path=${dataset}/val2017  \
         --bin_file_path=val2017_bin \
         --meta_file_path=val2017_bin_meta \
         --model_input_height=800  \
         --model_input_width=1216
   ```
   - 参数说明
      - --image_src_path：数据集路径
      - --bin_file_path：生成的图片bin文件路径
      - --meta_file_path：生成的图片附加信息路径（临时信息，get_info.py需要用到）
      每个图像对应生成一个二进制bin文件，一个附加信息文件。

3. 生成数据集info文件。

   执行“get_info.py”，会生成两个文件，其中“solo_meta.info”用于后处理。
   ```shell
   python get_info.py ${dataset}  \
         SOLO/configs/solo/solo_r50_fpn_8gpu_1x.py  \
         val2017_bin  \
         val2017_bin_meta  \
         solo.info  \
         solo_meta.info  \
         1216 800
   ```
   - 参数说明
      - ${dataset}：数据集路径。
      - SOLO/configs/solo/solo_r50_fpn_8gpu_1x.py：使用的开源代码文件路径。
      - val2017_bin：预处理后的数据文件的相对路径。
      - val2017_bin_meta：预处理后的数据文件的相对路径。
      - solo.info：生成的数据集文件保存的路径。
      - solo2_meta.info：生成的数据集文件保存的路径。
      - 1216：图片宽。
      - 800：图片高。

   运行成功后，在当前目录中生成“solov2_meta.info”。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      从源码包中获取权重文件：[SOLO_R50_1x.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/SOLOV1/PTH/SOLO_R50_1x.pth)，请将其放在与“solov1_pth2onnx.py”文件同一目录内。

   2. 导出onnx文件。

      1. 使用“SOLO_R50_1x.pth”导出onnx文件。

         运行“solov1_pth2onnx.py”脚本。

         ```shell
         python solov1_pth2onnx.py \
            --config SOLO/configs/solo/solo_r50_fpn_8gpu_1x.py \
            --pth_path SOLO_R50_1x.pth \
            --out SOLOv1.onnx \
            --shape 800 1216
         ```

         获得“SOLOv1.onnx”文件。

         - 参数说明：

           - --config：使用的开源代码文件路径。
           - ---pth_path：权重文件名称。
           - --out：输出文件名称。
           - --shape：图片参数。

      2. 优化ONNX模型。

         ```
         python -m onnxsim SOLOv1.onnx SOLOv1_sim.onnx
         ```

         获得SOLOv1_sim.onnx文件。

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

          ```shell
          atc --framework=5 \
               --model=SOLOv1_sim.onnx \
               --output=solo  \
               --input_format=NCHW \
               --input_shape="input:1,3,800,1216" \
               --log=error \
               --soc_version=Ascend${chip_name} \
               --customize_dtypes=./customize_dtypes.cfg \
               --precision_mode=force_fp16
          ```

          - 参数说明：

            -   --model：为ONNX模型文件。
            -   --framework：5代表ONNX模型。
            -   --output：输出的OM模型。
            -   --input_format：输入数据的格式。
            -   --input_shape：输入数据的shape。
            -   --log：日志级别。
            -   --soc_version：处理器型号。
            -   --customize_dtypes：自定义算子的计算精度。
            -   --precision_mode：其余算子的精度模式。
            运行成功后生成solo.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。
      > `output` 路径根据用户需求自由设置，这里以 `output=./out` 为例说明
      ```shell
      python -m ais_bench \
         --model "./solo.om" \
         --input "./val2017_bin/" \
         --output ${output} \
         --outfmt BIN \
         --device 0 \
         --batchsize 1 \
         --loop 1
      ```

      -   参数说明：

          -   --model：om文件路径。
          -   --input:输入路径
          -   --output：输出路径。
          -   --outfmt：输出数据的格式，默认”BIN“，可取值“NPY”、“BIN”、“TXT”。
          -   --loop：推理次数，可选参数，默认1，profiler为true时，推荐为1

         推理后的输出默认在 `--output` 文件夹下。


   3. 精度验证。

      调用脚本与数据集val2017标签比对。这里的 `dataset_path` 需要指定bin文件所在的路径，一般是 `dataset_path=${dataset}/2022_xxx` 这样的路径。
      ```shell
      python solov1_postprocess.py  \
         --dataset_path=${dataset_path}   \
         --model_config=SOLO/configs/solo/solo_r50_fpn_8gpu_1x.py  \
         --bin_data_path=${output}  \
         --meta_info=solo_meta.info \
         --net_out_num=3 \
         --model_input_height 800 \
         --model_input_width 1216
      ```

      - 参数说明：

        - --dataset_path：数据集路径。
        - --model_config：使用的开源代码文件路径。
        - --bin_data_path：推理结果所在目录。
        - --meta_info：数据预处理后获得的文件。
        - --net_out_num：输出节点数量。
        - --model_input_height：图片的高。
        - --model_input_width：图片的宽。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| Precision |       |
| --------- | ----- |
| 标杆精度  | 32.1% |
| 310精度   | 32.1% |
| 310P3精度 | 32.1% |

| Throughput | 310    | 310P3   | T4    | 310P3/310 | 310P3/T4 |
| ---------- | ------ | ------- | ----- | --------- | -------- |
| bs1        | 5.8629 | 10.5820 | 5.128 | 1.8049    | 2.009    |

该模型离线推理只支持bs1
