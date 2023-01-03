# Efficient-3DCNNs模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

  - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理精度&性能](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

3D-CNN网络在传统卷积网络的基础上引入了3D卷积，使得模型能够提取数据的时间维度和空间维度特征，从而能够完成更复杂的图像识别或者动作识别任务。Efficient-3DCNNs将高效的MobileNetV2与3D卷积结合，在动作识别领域取得了最好的水平。


- 参考论文：[Okan Köpüklü, Neslihan Kose, Ahmet Gunduz, Gerhard Rigoll. Resource Efficient 3D Convolutional Neural Networks.(2019)](https://arxiv.org/pdf/1904.02422.pdf))

- 参考实现：

  ```
  url= https://github.com/okankop/Efficient-3DCNNs
  branch=master 
  commit_id=d60c6c48cf2e81380d0a513e22e9d7f8467731af
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 大小                           | 数据类型 | 数据排布格式 |
  | -------- | ------------------------------ | -------- | ------------ |
  | input    | batchsize x 3 x 16 x 112 x 112 | RGB      | NCHW         |


- 输出数据

  | 输出数据 | 大小            | 数据类型 | 数据排布格式 |
  | -------- | --------------- | -------- | ------------ |
  | output1  | batchsize x 101 | FLOAT32  | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套       | 版本                               | 环境准备指导                                                 |
| ---------- | ---------------------------------- | ------------------------------------------------------------ |
| 固件与驱动 | 1.0.16（NPU驱动固件版本为5.1.RC2） | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN       | 5.1.RC2                            |                                                              |
| Python     | 3.7.5                              |                                                              |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```
   git clone https://github.com/okankop/Efficient-3DCNNs
   ```

2. 安装源码包中的依赖。

   ```
   pip install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型基于UCF-101训练和推理，UCF-101是一个轻量的动作识别数据汇集，包含101种动作的短视频。请用户自行获取UCF-101数据集，上传数据集到服务器任意目录并解压（如Efficient-3DCNNs/datasets）。

   数据集文件夹结构如下：

    ```
   UCF-101
       ├──dir1
           ├── video.avi
       ├──dir2
           ├── video.avi
    ```

2. 将视频提取为图片

   1. 在任意目录下载并解压ffmpeg（如：Efficient-3DCNNs/datasets）

      ```
      cd datasets
      wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz
      xz -d ffmpeg-git-amd64-static.tar.xz
      tar -xvf ffmpeg-git-amd64-static.tar
	  cd ..
      ```

   2. 将ffmpeg文件添加到/bin目录下。

      ```
      cd /bin 
      ln -s ${root_path}/datasets/ffmpeg-git-20210908-amd64-static/ffmpeg ffmpeg 
      ln -s ${root_path}/datasets/ffmpeg-git-20210908-amd64-static/ffprobe ffprobe
      cd -
      ```

      > 注意：ffmpeg-git-20210908-amd64-static名字会有所不同，需要自行更改。

   3. 在datasets目录下新建rawframes文件夹，运行Efficient-3DCNNs文件夹下的脚本将数据格式转为从视频帧中提取的图片。

      ```
      mkdir datasets/rawframes
      python Efficient-3DCNNs/utils/video_jpg_ucf101_hmdb51.py datasets/UCF-101/ datasets/rawframes 
      python Efficient-3DCNNs/utils/n_frames_ucf101_hmdb51.py datasets/rawframes
      ```

3. 数据预处理。

   数据预处理将图片数据转换为模型输入的二进制数据。

   执行Efficient-3DCNNs_preprocess.py脚本。需要根据不同的batchsize完成不同的数据预处理。

   ```
   cp Efficient-3DCNNs/annotation_UCF101/ucf101_01.json .
   python Efficient-3DCNNs_preprocess.py --video_path=datasets/rawframes --annotation_path=ucf101_01.json --output_path=bin_path --info_path=ucf101_bs1.info --inference_batch_size=1
   ```

   - 参数说明：
     - --video_path：为原始数据集（rawframes）路径。
	 - --annotation_path：数据集信息路径
     - --output_path：为模型输入数据（.bin）存储路径。
     - --inference_batch_size：batchsize的大小。
     - --info_path: 生成info文件的路径。

   执行成功后，在当前目录生成·bin_path，保存着预处理后的数据文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件 [ucf101_mobilenetv2_1.0x_RGB_16_best.pth](https://drive.google.com/drive/folders/1u4DO7kjAQP6Zdh8CN65iT5ozp11mvE-H?usp=sharing)，放在当前目录下。

   2. 导出onnx文件。

      使用ucf101_mobilenetv2_1.0x_RGB_16_best.pth导出onnx文件。

      运行Efficient-3DCNNs_pth2onnx.py脚本。

      ```
      python Efficient-3DCNNs_pth2onnx.py ucf101_mobilenetv2_1.0x_RGB_16_best.pth Efficient-3DCNNs.onnx
      ```

      - 参数说明：
        - ucf101_mobilenetv2_1.0x_RGB_16_best.pth：权重pth文件路径。
        - Efficient-3DCNNs.onnx：输出onnx文件路径。

      获得Efficient-3DCNNs.onnx文件。
    
   3. 简化onnx模型。

      ```
	   pip install onnx-simplifier
      python -m onnxsim Efficient-3DCNNs.onnx Efficient-3DCNNs_sim.onnx --input-shape "1,3,16,112,112" --dynamic-input-shape
      ```

      获得Efficient-3DCNNs_sim.onnx模型。

   4. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称(${chip_name})。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3
         回显如下：
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 17.6         57                0    / 0              |
         | 0       0         | 0000:3B:00.0    | 0            936 / 21534                            |
         +===================+=================+======================================================+
         ```

      3. 执行ATC命令。

         ```
         atc --framework=5 --model=Efficient-3DCNNs_sim.onnx --output=Efficient-3DCNNs_bs1 --input_format=NCHW --input_shape="image:1,3,16,112,112" --log=error --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

          运行成功后在onnx2om目录下生成 Efficient-3DCNNs_bs1.om 模型文件。



2. 开始推理验证。

   1. 使用ais_infer工具进行推理。
       ais-infer工具获取及使用方式请点击查看[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)。

   2. 执行推理。

      执行命令
      ```
      python -m ais_bench --model Efficient-3DCNNs_bs1.om --input bin_path --output result --output_dirname dumpout_bs1 --outfmt BIN --batchsize 1
      ```

      -   参数说明：
          - --model：om模型的路径。
          - --input：输入模型的二进制文件路径。
          - --output：推理结果输出目录。
          - --output_dirname：推理结果输出的二级目录名。
          - --batchsize：输入数据的batchsize。

      推理后的输出在当前目录result下。

      > **说明：** 执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见[参数详情](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer#参数说明)。


   3. 精度验证。

      调用Efficient-3DCNNs_postprocess.py脚本与数据集标签ucf101_01.json比对，可以获得Accuracy数据，保存在result_b1.json。

      ```
      python Efficient-3DCNNs_postprocess.py --result_path=result/dumpout_bs1 --info_path=ucf101_bs1.info --annotation_path=ucf101_01.json --acc_file=result_bs1.json
      ```
      - 参数说明：
         - --result_path：推理结果路径。
         - --info_path：预处理生成的数据集info文件。
         - --annotation_path：标签json文件。
         - --acc_file：推理结果的accuracy数据。
      获得result_bs1.json文件

    4. 性能验证。
       可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
       ```
       python -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
       ```
	   - 参数说明：
         - --model：om模型的路径。
         - --batchsize：数据集的batchsize。
         - --loop：循环的次数。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>


| 芯片型号 | Batch Size | 数据集  | 精度                           | 性能     |
| -------- | ---------- | ------- | ------------------------------ | -------- |
| 310P     | 1          | UCF-101 | top1:0.81073<br />top5:0.96325 | 1158.0105  |
| 310P     | 4          | UCF-101 | top1:0.81073<br />top5:0.96325 | 1245.1167  |
| 310P     | 8          | UCF-101 | top1:0.81073<br />top5:0.96325 | 1110.4093 |
| 310P     | 16         | UCF-101 | top1:0.81073<br />top5:0.96325 | 978.0638 |
| 310P     | 32         | UCF-101 | top1:0.81073<br />top5:0.96325 | 977.8292 |
| 310P     | 64         | UCF-101 | top1:0.81073<br />top5:0.96325 | 1009.593 |