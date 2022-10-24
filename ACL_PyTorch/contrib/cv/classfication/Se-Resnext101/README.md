
# SE-ResNeXt101 模型推理指导


## 1 模型概述

卷积神经网络CNN的核心是卷积操作，它通过融合局部感受野内的空间与通道信息来提取特征。大量已有的研究结果表明，提升特征层次结构中空间信息的编码，可以增强 CNN 的表示能力。在这项工作中，我们转而关注通道关系并提出了一种新颖的架构单元，我们将其称为 "Squeeze-and-Excitation"（SE）块，它通过显式构建通道之间的相互依赖关系来自适应地重新校准通道特征响应。将SE块堆叠形成 SENet 架构，可以有效地泛化不同的数据集。我们进一步证明，SE 块以很低的计算成本为现有最先进的 CNN 带来了显著的性能提升。以 Squeeze-and-Excitation Networks 为基础，我们在 ILSVRC 2017 分类竞赛中赢得冠军，并将 Top5 的错误率降低到 2.251%，相比 2016 年的获胜成绩提升了 25%。

**论文地址**：[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)  

**参考实现**：
+ 代码地址：[mmclassification: SE-ResNext101](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fopen-mmlab%2Fmmclassification%2Fblob%2Fmaster%2Fconfigs%2F_base_%2Fmodels%2Fseresnext101_32x4d.py)
+ tag：v0.23.0

**模型输入**：

| input-name | input-shape | data-type   | data-format |
| -------- | -------- | ---------- | ------------ |
| image | batchsize x 3 x 224 x 224 | RGB_FP32 | NCHW         |

**模型输出**：

| output-name | output-shape     | data-type | data-format |
| -------- | -------- | -------- | ------------ |
| class | 1 x 1000 | FLOAT32  | ND           |




## 2 环境说明
该模型离线推理使用 Atlas 300I Pro 推理卡，所有步骤都在 [CANN 5.1.RC2](https://www.hiascend.com/software/cann/commercial) 、Python 3.7.5 、PyTorch 1.12.1 环境下进行，CANN 包以及对应驱动、固件的安装请参考 [软件安装](https://www.hiascend.com/document/detail/zh/canncommercial/51RC2/envdeployment/instg)。所需的 Python 第三方依赖如下：

| 依赖库 | 版本|
| ---- | ----|
| torch | 1.12.1 |
| torchvision | 0.13.0 |
| numpy | 1.23.2 |
| mmcls | 0.23.2 |
| onnx | 1.12.0 |
| onnx-simplifier | 0.4.7 |
| Pillow | 9.2.0 |

## 3 快速上手

### 3.1 安装依赖
```shell
conda create -n senet python=3.7.5
conda activate senet
pip install -r requirements.txt
```

### 3.2 数据集预处理

1.  获取原始数据集
    
    本模型推理项目使用 ILSVRC2012 数据集验证模型精度，请在 [ImageNet官网](http://image-net.org/) 自行下载ILSVRC2012数据集并解压，本模型将用到 ILSVRC2012_img_val.tar 验证集及 ILSVRC2012_devkit_t12.gz 中的val_label.txt数据标签。
    
    请按以下的目录结构存放数据：
    ```
    ├──ImageNet/
        ├──ILSVRC2012_img_val/
            ├──ILSVRC2012_val_00000001.JPEG
            ├──ILSVRC2012_val_00000002.JPEG
            ├──...
        ├──ILSVRC2012_devkit_t12/
            ├──val_label.txt
    ```
2. 数据预处理
    ```python
    python3.7 Se_Resnext101_preprocess.py resnet /opt/npu/imageNet/val/ ./prep_dataset/
    ```
    参数说明：
    + resnet 该模型数据预处理方式同 ResNet 网络，所以此处设置为resnet。（脚本还支持inceptionv3和inceptionv4）
    + /opt/npu/imagenet/val/ 原始测试图片（.jpeg）所在目录的路径。
    + ./prep_dataset/ 指定一个目录用于存放生成的二进制（.bin）文件。
    
    运行成功后，每个图像对应生成一个二进制文件，存放于当前目录下的 prep_dataset 目录中。


### 3.3 模型转换


#### 3.3.1 PyTroch 模型转 ONNX 模型

1. 下载pth权重文件  
    - pth权重：[SE-ResNext101预训练pth权重文件](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E8%AE%AD%E7%BB%83/cv/image_classification/SE-resnext101/state_dict.pth) 
    - md5sum: 0C94EA7067268CB66D74001D8B01F7F8
    ```shell
    wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E8%AE%AD%E7%BB%83/cv/image_classification/SE-resnext101/state_dict.pth
    ```

2. 生成 ONNX 模型
    ```shell
    python3 Se_Resnext101_pth2onnx.py state_dict.pth se-resnext101.onnx
    ```

3. ONNX模型简化 
    ```shell
    python -m onnxsim se-resnext101.onnx se-resnext101-sim.onnx --overwrite-input-shape=16,3,224,224
    ```
    + 参数说明
      + --overwrite-input-shape：输入数据的排布格式
    
    至此，当前目录下会生成最终的ONNX模型： se-resnext101-sim.onnx 文件。

#### 3.3.2 ONNX 模型转 OM 模型

1. 设置环境变量
    ```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/Ascend/driver/lib64/driver
    ```
    说明：该脚本中环境变量仅供参考，请以实际安装路径配置环境变量。

2. 查看芯片名称（${chip_name}）
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

3. ONNX 模型转 OM 模型
    ```shell
    atc --framework=5 \
        --model=se-resnext101-sim.onnx \
        --output=se-resnext_bs16 \
        --input_format=NCHW \
        --input_shape="image:16,3,224,224" \
        --log=error \
        --soc_version=Ascend${chip_name}
    ```
    
   参数说明：
   
    - --model：为ONNX模型文件。
    - --framework：5代表ONNX模型。
    - --output：输出的OM模型。
    - --input_format：输入数据的排布格式。
    - --input_shape：输入数据的shape。
    - --log：日志级别。
    - --soc_version：处理器型号。

### 3.4 离线推理


#### 3.4.1 准备推理工具

+ 推理工具使用ais_infer，须自己拉取源码，打包并安装。
    ```shell
    # 指定CANN包的安装路径
    export CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest
    # 获取源码
    git clone https://gitee.com/ascend/tools.git
    cd tools/ais-bench_workload/tool/ais_infer/backend/
    # 打包，会在当前目录下生成 aclruntime-xxx.whl
    pip3 install --upgrade pip
    pip3 wheel ./
    # 安装
    pip3 install --force-reinstall aclruntime-xxx.whl
    ```
    参考：[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer#%E4%BB%8B%E7%BB%8D)

#### 3.4.2 执行推理
1. 需进入 ais_infer.py 所在目录执行以下代码
    ```shell
    # 对预处理后的数据进行推理
    cd ais_infer
    mkdir ../result/
    python3 ais_infer.py --model ../se-resnext_bs16.om --input ../prep_dataset/ --output ../result/ --outfmt TXT --batchsize 16
    cd ..
    ```
    参数说明：
    + --model：OM模型路径。
    + --input：存放预处理bin文件的目录路径
    + --output：存放推理结果的目录路径
    + --outfmt：推理输出文件的格式
    
    运行成功后，在--output指定的目录下，会生成一个根据执行开始时间来命名的子目录，用于存放推理结果文件。
2. 推理结束后需回到推理前所在的工作目录


#### 3.4.3  计算推理精度

1. 后处理计算精度
    ```shell
    python3.7 Se_Resnext101_postprocess.py \
        result/2022xxxxx/ \
        /opt/npu/imageNet/val_label.txt \
        ./result.json
    ```
    
    参数说明：
    
    +  result/2022xxxxx/：生成推理结果所在路径。
    +  /opt/npu/imageNet/val_label.txt：标签数据路径。
    +  ./result.json：结果文件保存路径。
    
2. 结果解读
   
    运行成功后，在当前目录下找到 result.json 文件，其内容为 top1 到 top5 的分类正确率：
    ```json
    {
        "title": "Overall statistical evaluation", 
        "value": [
            {"key": "Number of images", "value": "50000"}, 
            {"key": "Number of classes", "value": "1000"}, 
            {"key": "Top1 accuracy", "value": "78.24%"}, 
            {"key": "Top2 accuracy", "value": "87.99%"}, 
            {"key": "Top3 accuracy", "value": "91.32%"}, 
            {"key": "Top4 accuracy", "value": "93.14%"}, 
            {"key": "Top5 accuracy", "value": "94.23%"}
        ]
    }
    ```

#### 3.4.4  获取推理性能

1. 获取性能数据
  
    对于性能的测试，需要注意以下两点：
    + 测试前，请通过 npu-smi info  命令查看 NPU 设备状态，请务必在 NPU 设备空闲的状态下进行性能测试。
    + 为避免因测试持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    
    纯推理命令：
    ```shell
    python3 ais_infer.py --model ./se-resnext_bs16.om --batchsize 16 --loop 100
    ```
    执行完纯推理命令，程序会打印出跟性能先关的指标：
    ```log
    [INFO] -----------------Performance Summary------------------
    [INFO] H2D_latency (ms): min = 1.7399787902832031, max = 1.7399787902832031, mean = 1.7399787902832031, median = 1.7399787902832031, percentile(99%) = 1.7399787902832031
    [INFO] NPU_compute_time (ms): min = 21.843000411987305, max = 22.437000274658203, mean = 21.94005001068115, median = 21.921500205993652, percentile(99%) = 22.326120758056643
    [INFO] D2H_latency (ms): min = 0.1049041748046875, max = 0.1049041748046875, mean = 0.1049041748046875, median = 0.1049041748046875, percentile(99%) = 0.1049041748046875
    [INFO] throughput 1000*batchsize(16)/NPU_compute_time.mean(21.94005001068115): 729.2599603105127
    [INFO] ------------------------------------------------------
    ```

2. 计算性能

    使用吞吐率作为性能指标。
    > 吞吐率（throughput）： 模型在单位时间（1秒）内处理的数据样本数。
    
    计算吞吐率：
    + 执行纯推理时若指定了 batchsize，则找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率，本例中的吞吐率为 729.2599603105127
    + 若没有指定 batchsize，则可以通过 **NPU_compute_time** 中的 **mean** 来计算：
    $$throughput =\frac{batchsize}{mean} * 1000 =729.26(fps)$$


## 4 精度与性能对比

1. 精度对比

    自测了 batchsize 为 1 和 16 的精度，两个 batchsize 得到的精度没有差别，且与开源仓精度的相对误差小于 1%.
    <table>
    <tr>
    <th>Model</th>
    <th>batchsize</th>
    <th>Acc@Top1</th>
    <th>目标精度</th>
    <th>误差</th>
    </tr>
    <tr>
    <td rowspan="2">SE-ResNetXt101</td>
    <td>1</td>
    <td rowspan="2">78.24%</td>
    <td rowspan="2"><a href="https://github.com/open-mmlab/mmclassification/tree/master/configs/seresnet">78.26%</a></td>
    <td rowspan="2"> $$ \frac {|78.24-78.26|} {78.26}= 0.0003$$ </td>
    </tr>
    <tr>
    <td>16</td>
    </tr>
    </table>  

2. 性能对比
  
    在 310P 设备上，当 batchsize 为 4 时模型性能最优，达 903.506 fps.
    | batchsize | 310性能 | T4性能 | 310P性能 | 310P/310 | 310P/T4 |
    | ---- | ---- | ---- | ---- | ---- | ---- |
    | 1 | 182.224fps | 346.484 fps | 529.115fps | 2.904倍 | 1.527倍 |
    | 4 | 207.960 fps | 604.421 fps | 903.506fps | 4.345倍 | 1.495倍 |
    | 8 | 218.401 fps | 648.682 fps | 815.805fps | 3.735倍 | 1.258倍 |
    | 16 | 225.180 fps | 723.749 fps | 729.260fps | 3.238倍 | 1.008倍 |
    | 32 | 222.117 fps | 771.672 fps | 419.361 fps | 1.888倍 | 0.543倍 |
    | 64 | 223.843 fps | 789.784 fps | 603.140 fps | 2.694倍 | 0.764倍 |
    | 性能最优bs | 225.180 fps | 789.784 fps |903.506fps | 4.012倍 | 1.144倍 |

