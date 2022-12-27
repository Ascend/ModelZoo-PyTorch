# FCN-8s模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

FCN-8s定义并详细描述了全卷积网络的空间，解释了它们在空间密集预测任务中的应用，并绘制了与先验模型的连接。FCN-8s将当代分类网络（AlexNet、VGG网络和GoogLeNet）改编为完全卷积网络，并通过微调将其学习表示转移到分割任务。然后，FCN-8s定义了一种新的架构，该架构将来自深层粗层的语义信息与来自浅层细层的外观信息相结合，以产生精确和详细的分割。FCN-8s的全卷积网络实现了PASCAL VOC（与2012年的平均IU 62.2%相比，相对提高了20%）、NYUDv2和SIFT流的最先进分割，而对于典型图像，推理需要三分之一秒。


- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmsegmentation.git
  branch=master
  commit_id=e6a8791ab0a03c60c0a9abb8456cd4d804342e92
  model_name=FCN-8s
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
  | input    | RGB_FP32 | 1 x 3 x 500 x 500 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型  | 大小 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output  |  INT32 |1 x 1 x 500 x 500 | ND           |



# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

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
   pip3 install -r requirements.txt
   ```

   ```
   pip3 install mmcv-full==1.6.1
   ```
   
   注：此步需要较长时间


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型该模型使用VOC2012的1449张验证集进行测试，数据集在服务器的目录为/opt/npu/VOCdevkit/VOC2012，若服务器没有此数据集，请用户自行获取该数据集，上传并解压数据集到服务器任意目录。（如：/home/HwHiAiUser/datasets）

   ```
   ├── VOC2012
          └── Annotations
          └── ImageSets
                  └── Segmentation
                           └── val.txt
   ├── JPEGImages
          └── voc12_jpg.info
   ├── SegmentationClass
   ├── SegmentationObject      
   ```

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   执行FCN-8s_preprocess.py脚本，完成预处理。

   ```
   python3.7 FCN-8s_preprocess.py --image_folder_path=/opt/npu/VOCdevkit/VOC2012/JPEGImages/ --split=/opt/npu/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt --bin_folder_path=./voc12_bin/
   ```
    -   参数说明：

        -   image_folder_path：原始数据验证集（.jpeg）所在路径。
        -   bin_folder_path：输出的二进制文件（.bin）所在路径。

3. 运行get_info.py脚本，生成图片数据info文件。

   数据预处理将原始数据集转换为模型输入的数据。

   执行get_info.py脚本，完成预处理。

   ```
   python3.7 get_info.py jpg /opt/npu/VOCdevkit/VOC2012/JPEGImages/ voc12_jpg.info
   ```

    -   参数说明：

        -   第一个参数：生成的数据集文件格式。
        -   第二个参数：预处理后的数据文件相对路径。
        -   第三个参数：生成的数据集文件名。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       获取fcn-8s基于mmsegmentation预训练的npu权重文件，下载链接：[fcn_r50-d8_512x512_20k_voc12aug_20200617_010715-52dc5306.pth](https://download.openmmlab.com/mmsegmentation/v0.5/fcn/fcn_r50-d8_512x512_20k_voc12aug/fcn_r50-d8_512x512_20k_voc12aug_20200617_010715-52dc5306.pth)

       ```
       wget https://download.openmmlab.com/mmsegmentation/v0.5/fcn/fcn_r50-d8_512x512_20k_voc12aug/fcn_r50-d8_512x512_20k_voc12aug_20200617_010715-52dc5306.pth
       ```

   2. 导出onnx文件。

      1. mmsegmentation源码安装

         ```
         git clone https://github.com/open-mmlab/mmcv.git
		 cd mmcv
		 pip install -e .
		 cd ..
		 git clone https://github.com/open-mmlab/mmsegmentation.git
		 cd mmsegmentation
		 pip install -e . 
		 cd ..
         ```

      2. 使用pytorch2onnx.py导出onnx文件。

         运行pytorch2onnx.py脚本导出指定batch size为1的onnx模型，模型不支持动态batch。

         ```
         python3.7 mmsegmentation/tools/pytorch2onnx.py mmsegmentation/configs/fcn/fcn_r50-d8_512x512_20k_voc12aug.py --checkpoint fcn_r50-d8_512x512_20k_voc12aug_20200617_010715-52dc5306.pth --output-file fcn_r50-d8_512x512_20k_voc12aug.onnx --shape 500 500 --show
         ```

         获得fcn_r50-d8_512x512_20k_voc12aug.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
          source /opt/npu/CANN-6.1.1/ascend-toolkit/set_env.sh
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
         | 0       310P3     | OK              | 17.4         68                0    / 0              |
         | 0       0         | 0000:3B:00.0    | 0            930 / 21534                             |
         +===================+=================+======================================================
         ```

      3. 执行ATC命令。

         ```
         atc --framework=5 --model=fcn_r50-d8_512x512_20k_voc12aug.onnx  --output=fcn_r50-d8_512x512_20k_voc12aug_bs1 --input_format=NCHW --input_shape="input:1,3,500,500" --log=debug --soc_version=${chip_name}
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

           运行成功后生成<u>fcn_r50-d8_512x512_20k_voc12aug_bs1.om***</u>模型文件。



2. 开始推理验证。

   a.  安装ais_bench推理工具。
      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

   b.  执行推理。

      ```
      mkdir result
      ```

      ```
      python3.7 -m ais_bench --device 0 --batchsize 1 --model ./fcn_r50-d8_512x512_20k_voc12aug_bs1.om --input ./voc12_bin/ --output ./result/
      ```

      -   参数说明：

          -   --model：om模型路径
          -   --input：预处理后的输入数据。
          -   --batchsize：om模型的batchsize。
          -   --output：推理结果存放目录。

          推理后的输出在目录“./result/Timestam”下，Timestam为日期+时间的子文件夹,如 2022_08_24-16_16_28


   c.  精度验证。

      调用FCN-8s_postprocess.py评测bs1的mIoU精度：

      ```
    python3.7 FCN-8s_postprocess.py --bin_data_path=./result/2022_08_24-16_16_28 --test_annotation=./voc12_jpg.info --img_dir=/opt/npu/VOCdevkit/VOC2012/JPEGImages --ann_dir=/opt/npu/VOCdevkit/VOC2012/SegmentationClass --split=/opt/npu/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt --net_input_width=500 --net_input_height=500
      ```

      -   参数说明：

          -   --bin_data_path：推理结果
          -   --test_annotation：原始图片信息文件
          -   --img_dir：原始图片位置
          -   --ann_dir：验证图片位置
          -   --spli：图片的split
          -   --net_input_width：网宽
          -   --net_input_height：网高

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号   | Batch Size | 数据集    | 精度                                 | 性能            |
| --------- | ---------- | -------- | ----------------------------------- | --------------- |
|Ascend310P |    1       |  VOC2012 |mIoU：69.01% mAcc：78.94% aAcc：93.04%|  91.5592        |
|Ascend310  |    1       |  VOC2012 |mIoU：69.01% mAcc：78.94% aAcc：93.04%|  64.8908        |
|T4         |    1       |    -     |         -                           |  58.6830       |