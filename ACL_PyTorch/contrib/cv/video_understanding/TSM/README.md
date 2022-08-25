# TSM模型-推理指导


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

    [TSM论文](https://arxiv.org/abs/1811.08383)  
    TSM是一种通用且有效的时间偏移模块，它具有高效率和高性能，可以在达到3D CNN性能的同时，保持2D CNN的复杂性。TSM沿时间维度移动部分通道，从而促进相邻帧之间的信息交换。TSM可以插入到2D CNN中以实现零计算和零参数的时间建模。TSM可以扩展到在线设置，从而实现实时低延迟在线视频识别和视频对象检测。



- 参考实现：

    [mmaction2框架TSM代码](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm)




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
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.9.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>



1. 安装依赖。

    | 依赖名 | 版本号 |
    | :------: | :------: |
    | CANN  | 5.1.RC2 |
    | CANN（仅在atc转换OM时）  | 5.0.3 / 5.1.RC2 |
    | CANN（除了使用atc以外的实验步骤时）  | 5.0.3 / 5.0.4 / 5.1.RC2 |
    | python  | ==3.7.5 |
    | torch   | ==1.9.0 (cpu版本即可) |
    | onnx  | ==1.9.0 |
    | torchvision | ==0.10.0 |
    | numpy  | ==1.21.0 |
    | mmcv  | ==1.3.9 |
    | opencv-python  | ==4.5.3.56 |    




    **说明：** 
    >   X86架构：opencv,pytorch,torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
    >
    >   Arm架构：opencv,pytorch,torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装


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

2. 数据预处理。



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

    2. mmaction2源码安装
        ```shell
        git clone https://github.com/open-mmlab/mmaction2.git
        cd mmaction2
        pip install -r requirements/build.txt
        pip install -v -e .
        cd ..
        ```

    **说明：**  
    > 安装所需的依赖说明请参考mmaction2/docs/install.md

    3. 转换onnx
        ```shell
        python TSM_pytorch2onnx.py mmaction2/configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb.py ./tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb_20210630-1fae312b.pth --output-file=tsm.onnx --softmax --verify --show --shape 1 8 3 224 224
        ```

    4. 简化onnx
        使用onnxsim去除atc工具不支持的pad动态shape
        ```shell
        python3.7 -m onnxsim --input-shape="1,8,3,224,224" tsm.onnx onnx_sim/tsm_bs1.onnx
        ```
        若要获得不同batch_size的简化模型，只需要修改--input-shape参数，例如batch_size=16
        ```shell
        python3.7 -m onnxsim --input-shape="16,8,3,224,224" tsm.onnx onnx_sim/tsm_bs16.onnx
        ```

   5. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

            ```shell
            export install_path=/usr/local/Ascend/ascend-toolkit/latest
            export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
            export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
            export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
            export ASCEND_OPP_PATH=${install_path}/opp
            export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/
            export REPEAT_TUNE=True
            ```
            上述环境变量可通过运行脚本添加
            ```
            source env.sh
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
      2. 使用atc将onnx模型转换为om模型文件
       
      3. 执行ATC命令。

            ```shell
            atc --model=onnx_sim/tsm_bs1.onnx --framework=5 --output=om/tsm_bs1 --input_format=NCDHW --input_shape="video:1,8,3,224,224" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"
            ```

            若需要获取不同batch_size输入的om模型，则可以通过设定input_shape进行指定。下面的命令可生成batch_size=16的模型。

            ```shell
            atc --model=onnx_sim/tsm_bs16.onnx --framework=5 --output=om/tsm_bs16 --input_format=NCDHW --input_shape="video:16,8,3,224,224" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"
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




2. 开始推理验证。<u>***根据实际推理工具编写***</u>

a.  使用ais-infer工具进行推理。

   执行命令增加工具可执行权限，并根据OS架构选择工具

    下载ais_infer 推理工具代码

    ```
    git clone https://gitee.com/ascend/tools.git
    ```
    
    进入ais-bench/tool/ais_infer目录下执行如下命令进行编译，即可生成推理后端whl包

    ```
    cd tools/ais-bench_workload/tool/ais_infer/backend/
    pip3 wheel ./
    ```
    在运行设备上执行如下命令，进行安装

    pip3 install ./aclruntime-0.0.1-cp37-cp37m-linux_x86_64.whl

    如果安装提示已经安装了相同版本的whl，请执行命令请添加参数"--force-reinstall"


b.  执行推理。

    ```
     python3 ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model ./tsm_bs1.om --input "./ucf101/out_bin_1" --batchsize 1 --output ./output/out_bs1/
    ```

    -   参数说明：

        -   model：om文件路径。
        -   input：数据集路径。
        -   batchsize：1。
        -   output:输出路径
		...

        >**说明：** 
        >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

c.  精度验证。

    执行后处理脚本，获取精度

    ```shell
    python TSM_postprocess.py --result_path ./output/out_bs1 --info_path /opt/npu/ucf101/ucf101.info
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
    python TSM_postprocess.py --result_path ./output/out_bs16/20210727_143344/ --info_path /opt/npu/ucf101/ucf101.info
    ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| Throughput | 310     | 310P    | T4     | 310P/310    | 310P/T4     |
| ---------- | ------- | ------- | ------ | ----------- | ----------- |
| bs1        |24.8 | 171.04 | 98.01	| 7.16|1.81|
| bs4        |22.48|132.23|107.90|5.88|1.22|
| bs8        |20.25|123.814|100.0|6.11|1.23|
| bs16       | 19.86|119.71|101.89|6.02|1.17|
| bs32       | 18.90|99.78|100.91|5.27|0.98|
|            |         |         |        |             |             |
| 最优batch 1  | 24.8|177.79|107.90|7.16|1.64 |