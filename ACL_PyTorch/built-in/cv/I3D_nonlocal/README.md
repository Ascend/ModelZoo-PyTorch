# I3D模型PyTorch离线推理指导
- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

​I3D是一种新的基于2D ConvNet 膨胀的双流膨胀3D ConvNet (I3D)。一个I3D网络在RGB输入上训练，另一个在流输入上训练，这些输入携带优化的、平滑的流信息。 模型分别训练了这两个网络，并在测试时将它们的预测进行平均后输出。深度图像分类ConvNets的过滤器和池化内核从2D被扩展为3D，从而可以从视频中学习效果良好的时空特征提取器并改善ImageNet的架构设计，甚至是它们的参数。[论文链接](https://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html)

- 参考实现
```
url=https://github.com/open-mmlab/mmaction2
```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FLOAT32 | batch x 10 x 3 x 32 x 256 x 256 | batch x clip x channel x time x height x width         |

   若想使用其他维度大小的输入，请修改i3d_pth2onnx.sh和i3d_onnx2om.sh文件

- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | --------| -------- | -------- | ------------ |
  | output  | FLOAT32  | batchsize x 10 x 400 | ND           | 


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | [CANN推理架构准备](https://www/hiascend.com/software/cann/commercial) |
  | Python                                                       | 3.7.5   | 创建anaconda环境时指定python版本即可，conda create -n ${your_env_name} python==3.7.5 |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码：

    ```sh
    git clone https://github.com/open-mmlab/mmaction2.git
    cd mmaction2
    git checkout dbf5d59fa592818325285b786a0eab8031d9bc80
    cd ..
    ```

    打Patch，修改代码
    ```
    patch -p0 < i3d_nonlocal.patch
    ```

    然后在仓库中创建目录。
    ```sh
    cd mmaction2
    mkdir -p data/kinetics400
    cd ..
    ```

2. 安装依赖，测试环境时可能已经安装其中的一些不同版本的库，故手动测试时不推荐使用该命令安装

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集：运行仓库中的tools/data/kinetics/download_backup_annotations.sh.将会在data/kinetics400目录下创建annotations目录。

    ```
    cd ./mmaction2/tools/data/kinetics
    bash download_backup_annotations.sh kinetics400
    cd ../../..
    ```

    最后获取kinetics400验证集。

    |   数据集    | 验证集视频 |
    | :---------: | :--------: |
    | kinetics400 |   19796    |

    我们使用的 Kinetics400 验证集包含 19796 个视频，用户可以从 [验证集视频](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB) 下载这些视频。
    下载并解压好后，将数据集重命名为videos_val并放置于data/kinetics400目录下。

    数据集准备完成。

2. 数据预处理

    若有另外需求，请参考[数据处理](docs_zh_CN/data_preparation.md)。

    运行预处理脚本。将对videos_val中的所有视频进行抽帧处理，并将结果放置在data/kinetics400/rawframes_val目录下。本脚本采用Opencv对mp4格式的视频，采用4线程抽取256*256大小的RGB帧，输出格式为jpg。

    ```
    python3 tools/data/build_rawframes.py data/kinetics400/videos_val data/kinetics400/rawframes_val --task rgb --level 1 --num-worker 4 --out-format jpg --ext mp4 --new-width 256 --new-height 256 --use-opencv
    ```

    运行该脚本获取验证所需要的验证文件。将生成kinetics400_label.txt和kinetics400_val_list_rawframes.txt。kinetics400_val_list_rawframes.txt即为验证时需要的文件。

    ```sh
    python3 data/kinetics400/generate_labels.py
    cd ..
    ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换

    1. 获取权重文件。
        ```
        wget https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb_20200814-7c30d5bb.pth
        ```
        首先将该权重文件重命名为i3d_nl_dot_product_r50，并在mmaction2目录下创建新目录checkpoints，并保存在checkpoints目录下。
        ```
        mkdir -p ./mmaction2/checkpoints
        mv i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb_20200814-7c30d5bb.pth ./mmaction2/checkpoints/i3d_nl_dot_product_r50.pth
        ```

    2. 导出onnx文件

        1. 使用mmaction2/tools目录下的pytorch2onnx.py导出onnx文件。
           运行pytorch2onnx.py脚本
           ```
           cd mmaction2
           python3 tools/deployment/pytorch2onnx.py configs/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb.py checkpoints/i3d_nl_dot_product_r50.pth --shape 1 10 3 32 256 256 --verify --show --output i3d.onnx --opset-version 11
           ```
           
           - 参数说明：
            -   --shape：格式为 batch clip channel time height width。
            -   --verify：对比onnx的输出与pth的输出，默认为false。
            -   --show：显示模型计算图，默认为false。
            -   --output：输出onnx模型文件名。

        2. 使用onnxsimplifier对模型进行简化
           ```
           python3 -m onnxsim i3d.onnx i3d_sim.onnx
           ```

    3. 使用ATC工具将onnx模型转为om模型

        1. 配置环境变量
            ```sh
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

        
        3. 执行ATC命令

            本节采用的模型输入为:1x10x3x32x256x256.（`$batch $clip $channel $time $height $width` ）。实验证明，若想提高模型精度，可增加`$clip`的值，但性能会相应降低。若想使用其他维度大小的输入，请修改i3d_pth2onnx.sh和i3d_onnx2om.sh文件。由于本模型较大，选择Ascend310的话batch_size只能设置为1，若大于1则会因为 Ascend310 内存不足而报错；选择Ascend310P3的话batch_size可以设置为1，4，8。
            ```
            atc --framework=5 --output=./i3d_bs1  --input_format=NCHW  --soc_version=Ascend${chip_name} --model=./i3d_sim.onnx --input_shape="0:1,10,3,32,256,256"
            ```

            - 参数说明：
            -   --framework：5代表ONNX模型。
            -   --model：为ONNX模型文件。
            -   --input\_format：输入数据的格式。
            -   --input\_shape：输入数据的shape。
            -   --output：输出的OM模型（bs后的数字为batchsize的大小）。           
            -   --soc\_version：处理器型号。

            运行成功后生成<u>***i3d_bs1.om***</u>模型文件。

2. 开始推理验证。

    1. 使用ais_bench工具进行推理。

        ais_bench工具获取及使用方式请点击查看[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)

    2. 精度验证。

        调用脚本i3d_infer.sh，即可获取top1_acc，top5_acc和mean_acc。
        ```
        mv ../i3d_inference.py ./
        python i3d_inference.py ./configs/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb.py --eval top_k_accuracy mean_class_accuracy --out result.json --batch_size 1 --model ./i3d_bs1.om --device_id 0
        ```

    3. 性能验证。

        可使用ais_bench推理工具的纯推理模型验证不同batch_size的om模型的性能，参考命令如下：

        ```
        cd ..
        python3 -m ais_bench --model=i3d_bs1.om
        ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据，由于本模型较大，310P3只测试batch_size为1，4，8情况下的精度与性能

| 芯片型号 | Batch Size | 数据集|  精度TOP1 | 精度TOP5 | 性能|
| --------- | ----| ----------| ------     |---------|---------|
| 310P3 |  1       | kinetics400 |   70.03     |  89.51  |  14.39     |
| 310P3 |  4       | kinetics400 |   70.03    |   89.51 |  14.28     |
| 310P3 |  8       | kinetics400 |   70.03    |   89.51 |  14.15   |
