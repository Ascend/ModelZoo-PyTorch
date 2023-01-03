#  PointRend 模型-推理指导

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

PointRend模型的作者提出了一种新的方法来从高质量的图像中进行物体和场景的分割而这种方法借鉴了传统的计算机图形学用于渲染高清图片的处理技术。PointRend的本质是一个新型的神经网络模块，能够通过一种不断迭代的算法来自适应地挑选出有问题的区域，并对该区域地像素点进行精细化地调整预测。除了作者使用的模型结构，PointRend模块也能够很灵活地被运用于现存的各个实例与语义分割模型上。

- 参考论文：

  [Alexander Kirillov, Yuxin Wu, Kaiming He, Ross Girshick.PointRend: Image Segmentation as Rendering.(2019)](http://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html)

- 参考实现：

  ```
  url=https://github.com/facebookresearch/detectron2
  commit_id=861b50a8894a7b3cccd5e4a927a4130e8109f7b8
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 1024 2048 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小                       | 数据排布格式 |
  | -------- | -------- | -------------------------- | ------------ |
  | output1  | RGB_FP32 | batchsize x 19 x 1024 2048 | NCHW         |

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

2. 在同级目录下获取开源模型代码并使用补丁文件。

   ```
   git clone https://github.com/facebookresearch/detectron2
   cd detectron2
   git reset --hard 861b50a8894a7b3cccd5e4a927a4130e8109f7b8
   patch -p1 < ../PointRend.diff
   cd ..
   python3.7 -m pip install -e detectron2
   ```
   
3. 安装必要依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   该模型在Cityscapes数据集上进行推理验证。从[官网](https://gitee.com/link?target=https%3A%2F%2Fwww.cityscapes-dataset.com%2F)获取`gtFine_trainvaltest.zip`和`leftImg8bit_trainvaltest.zip`，将这两个压缩包解压到创建的任意目录下。数据集目录结构如下所示：

   ```
   cityscapes
   ├── gtFine
   │   ├── test
   │   ├── train
   │   ├── val
   ├── leftImg8bit
   │   ├── test
   │   ├── train
   │   ├── val
   ```

3. 数据预处理，将原始数据集转换为模型的输入数据。

   执行 PointRend_preprocess.py 脚本，完成数据预处理。

   ```
   python3 PointRend_preprocess.py ${data_dir} ${save_dir}
   ```

   参数说明：

   - --data_dir：原数据集所在路径。
   - --save_dir：生成数据集二进制文件。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      点击获取权重文件[model_final_cf6ac1.pkl](https://dl.fbaipublicfiles.com/detectron2/PointRend/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes/202576688/model_final_cf6ac1.pkl)。

   2. 导出onnx文件。

      1. 使用PointRend_pkl2onnx.py导出batch_size=1的onnx文件。

         ```
         python3 PointRend_pkl2onnx.py detectron2/projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml ./model_final_cf6ac1.pkl ./PointRend.onnx 
         ```

         参数说明：

         - --参数1：模型配置文件。
         - --参数2：模型权重文件。
         - --参数3：输出的onnx文件。

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
      atc  --model=./PointRend.onnx  --framework=5 --output=PointRend_bs1 --input_format=NCHW  --input_shape="images:1,3,1024,2048" --log=error  --soc_version=${chip_name}
      ```
      
      运行成功后生成PointRend_bs1.om模型文件。

      参数说明：

      - --model：为ONNX模型文件。
      - --framework：5代表ONNX模型。
      - --output：输出的OM模型。
      - --input_format：输入数据的格式。
      - --input_shape：输入数据的shape。
      - --log：日志级别。
      - --soc_version：处理器型号。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      ```
      python3 -m ais_bench --model ./PointRend_bs1.om --input ${save_dir} --output result --output_dirname result_bs1 --outfmt BIN
      ```
      
      参数说明：
      
      - --model：om模型路径。
      - --input：输入数据所在路径。
      - --output：推理结果输出路径。
      - --output_dirname：推理结果输出子文件夹。
      - --outfmt：推理结果输出格式
   
3. 精度验证。

   调用脚本与数据集真值标签比对以获得精度。

   ```
   python3 PointRend_postprocess.py ${data_dir} ${result_dir}
   ```

   参数说明：

   - --data_dir：数据集路径。
   - --result_dir：推理结果所在路径，这里为 ./result/result_bs1。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，PointRend模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集     | 开源精度（mIoU）                                             | 参考精度（mIoU） |
| ----------- | ---------- | ---------- | ------------------------------------------------------------ | ---------------- |
| Ascend310P3 | 1          | Cityscapes | [78.9%](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend#semantic-segmentation) | 78.85%           |

| 芯片型号    | Batch Size | 参考性能（FPS） |
| ----------- | ---------- | --------------- |
| Ascend310P3 | 1          | 1.27            |