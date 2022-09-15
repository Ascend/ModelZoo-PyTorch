# TSM模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  <!-- ****** -->




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

  TSM是一种通用且有效的时间偏移模块，它具有高效率和高性能，可以在达到3D CNN性能的同时，保持2D CNN的复杂性。TSM沿时间维度移动部分通道，从而促进相邻帧之间的信息交换。TSM可以插入到2D CNN中以实现零计算和零参数的时间建模。TSM可以扩展到在线设置，从而实现实时低延迟在线视频识别和视频对象检测。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmaction2.git
  branch=master
  commit_id=5fa8faa
  model_name=configs/recognition/tsm
  ```




  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone https://github.com/open-mmlab/mmaction2.git       # 克隆仓库的代码
  cd /mmaction2              # 切换到模型的代码仓目录
  git checkout master         # 切换到对应分支
  git reset --hard 5fa8faa      # 代码设置到对应的commit_id（可选）
  cd /configs/recognition/tsm                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```






## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 8 x 3 x 224 x 224 | NTCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output_0  | 1 x 101 | FLOAT32  | ND           |



# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| python  | 3.7.5 |-|
| pytorch   | 1.9.0|-|
| pytorch   | 1.9.0|-|
| onnx  | 1.9.0 |-|
| torchvision | 0.10.0 |-|
| numpy  | 1.21.0 |-|
| mmcv  | 1.3.9 |-|
| opencv-python  | 4.5.3.56 |-|
| 操作系统  | 18.04.1 |-|
# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

    该模型使用 [UCF-101](https://www.crcv.ucf.edu/research/data-sets/ucf101/) 的验证集进行测试，数据集下载步骤如下
    ```shell
    cd ./mmaction2/tools/data/ucf101
    bash download_annotations.sh
    bash download_videos.sh
    bash extract_rgb_frames_opencv.sh
    bash generate_videos_filelist.sh
    bash generate_rawframes_filelist.sh
    ```
    （可选）本项目默认将数据集存放于/opt/npu/
    ```
    opt
    └── npu
        └── ucf101
    ```
2. 安装必要环境
    ```shell
    pip install -r requirements.txt  
    ```  
    mmaction2源码安装
    ```shell
    git clone https://github.com/open-mmlab/mmaction2.git
    cd mmaction2
    pip install -r requirements/build.txt
    pip install -v -e .
    cd ..
    ```

    **说明：**  
    > 安装所需的依赖说明请参考mmaction2/docs/install.md
3. 数据预处理。


    执行预处理脚本，生成数据集预处理后的bin文件以及相应的info文件
    ```shell
    python TSM_preprocess.py --batch_size 1 --data_root /opt/npu/ucf101/rawframes/ --ann_file /opt/npu/ucf101/ucf101_val_split_1_rawframes.txt --name out_bin_1
    ```
    第一个参数为batch_size，第二个参数为图片所在路径，第三个参数为图片对应的信息（由bash generate_rawframes_filelist.sh生成），第四个参数为保存的bin文件和info文件所在目录的名称。

    下面的命令可生成batch_size=16的数据文件以及相应的info文件
    ```shell
    python TSM_preprocess.py --batch_size 16 --data_root /opt/npu/ucf101/rawframes/ --ann_file /opt/npu/ucf101/ucf101_val_split_1_rawframes.txt --name out_bin_16
    ```

    若数据集下载位置不同，请将数据集目录（/opt/npu/ucf101）替换为相应的目录，若按照上述步骤下载数据集至./mmaction2/tools/data/ucf101，则无需指定这两个目录参数。

    预处理后的bin文件默认保存于{数据集目录}/{name}/，info文件保存为{数据集目录}/ucf101.info。

    若需要测试不同batch_size下模型的性能，可以指定batch_size以及保存目录的名称name。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

    1. 下载pth权重文件
        [TSM基于mmaction2预训练的权重文件](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb_20210630-1fae312b.pth)
        ```
        wget https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb_20210630-1fae312b.pth
        ```



    2. 转换onnx
        ```shell
        python TSM_pytorch2onnx.py mmaction2/configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb.py ./tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb_20210630-1fae312b.pth --output-file=tsm.onnx --softmax --verify --show --shape 1 8 3 224 224
        ```

    3. 简化onnx
        使用onnxsim去除atc工具不支持的pad动态shape
        ```shell
        mkdir onnx_sim
        python3.7 -m onnxsim --input-shape="1,8,3,224,224" tsm.onnx onnx_sim/tsm_bs1.onnx
        ```
        若要获得不同batch_size的简化模型，只需要修改--input-shape参数，例如batch_size=16
        ```shell
        mkdir onnx_sim
        python3.7 -m onnxsim --input-shape="16,8,3,224,224" tsm.onnx onnx_sim/tsm_bs16.onnx
        ```

    4. 使用ATC工具将ONNX模型转OM模型。

        1. 配置环境变量。

              ```
              source /opt/npu/CANN-RC2/ascend-toolkit/set_env.sh  
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
        3. 使用atc将onnx模型转换为om模型文件

              ```shell
              atc --model=onnx_sim/tsm_bs1.onnx --framework=5 --output=tsm_bs1 --input_format=NCDHW --input_shape="video:1,8,3,224,224" --log=debug --soc_version=${chip_name}
              ```

              若需要获取不同batch_size输入的om模型，则可以通过设定input_shape进行指定。下面的命令可生成batch_size=16的模型。

              ```shell
              atc --model=onnx_sim/tsm_bs16.onnx --framework=5 --output=tsm_bs16 --input_format=NCDHW --input_shape="video:16,8,3,224,224" --log=debug --soc_version=${chip_name}
              ```  

          - 参数说明：

            -   --model：为ONNX模型文件。
            -   --framework：5代表ONNX模型。
            -   --output：输出的OM模型。
            -   --input\_format：输入数据的格式。
            -   --input\_shape：输入数据的shape。
            -   --log：日志级别。
            -   --soc\_version：处理器型号。




2. 开始推理验证。

    1. 使用ais_infer工具进行离线推理.

        ```
        mkdir output
        cd output
        mkdir out_bs1
        cd ../
        python3 ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model ./om/tsm_bs1.om --input "./ucf101/out_bin_1" --batchsize 1 --output ./output/out_bs1/ --outfmt TXT
        ```
        **参数说明:**
        - --mode：om模型路径。
        - --input：二进制数据集文件夹路径。
        - --output：输出文件夹路径。
        - --outfmt：后处理输出格式。
        - --batchsize：推理的batchsize大小。

    2. 精度验证


        执行后处理脚本，获取精度
        ```
        python TSM_postprocess.py --result_path ./output/out_bs1/{result_name} --info_path ./ucf101/ucf101.info
        ```
        第一个参数为预测结果所在路径（需根据实际输出路径进行修改），第二个参数为数据集info文件路径

        当batch_size=1时，执行完成后，程序会打印出精度：
        ```
        Evaluating top_k_accuracy ...

        top1_acc	0.9448
        top5_acc	0.9963
        ```

        下面的命令可以针对batch_size=16的情况计算精度
        ```shell
        python TSM_postprocess.py --result_path ./output/out_bs16/20210727_143344/ --info_path ./ucf101/ucf101.info
        ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

  调用ACL接口推理计算，性能参考下列数据。

## 性能
| Model     | Batch Size | 310 (FPS/Card) | 710 (FPS/Card) | T4 (FPS/Card) | 710/310   | 710/T4      |
| --------- | ---------- | -------------- | -------------- | ------------- | --------- | ----------- |
| TSM   | 1          | 24.80 | 171.04 | 98.01	| 7.16|1.81|
| TSM   | 4          | 22.48|132.23|107.90|5.88|1.22|
| TSM   | 8          | 20.25|123.814|100.0|6.11|1.23|
| TSM   | 16         | 19.86|119.71|101.89|6.02|1.17|
| TSM   | 32         | 18.90|99.78|100.91|5.27|0.98|
| 最优batch |    1      | 24.8|177.79|107.90|7.16|1.64 |

## 精度

|Batch_size	| Framework	|  Container	| Precision |	Dataset |	Accuracy |	Ascend AI Processor |
| --------- | ---------- |  ------------- | --------- | ----------- |------------- | --------- |
| 1 |	PyTorch	| 	NA |	fp16 |	UCF101 |	top1:0.9402 top5:0.9958 |	Ascend 310 |
| 1 |	PyTorch	| 	NA |	fp16 |	UCF101 |	top1:0.9402 top5:0.9958 |	Ascend 310P |
