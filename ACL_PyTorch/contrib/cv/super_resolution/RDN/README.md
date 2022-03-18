## RDN Onnx模型PyTorch端到端推理指导

### 1 模型概述

#### 1.1 论文地址

[RDN论文](https://arxiv.org/abs/1802.08797)



#### 1.2 代码地址

[RDN代码](https://github.com/yjn870/RDN-pytorch)



### 2 环境说明

#### 2.1 深度学习框架

```
CANN 5.0.2
pytorch = 1.5.0
torchvision = 0.6.0
onnx = 1.7.0
```



#### 2.2 python第三方库

```
numpy == 1.20.3
Pillow == 8.2.0
opencv-python == 4.5.2.54
mmcv == 1.3.12
```

**说明：**

> X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
> Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装



### 3 模型转换

#### 3.1 pth转onnx模型

1. 下载pth权重文件

   [RDN_x2预训练pth权重文件](https://www.dropbox.com/s/pd52pkmaik1ri0h/rdn_x2.pth?dl=0)

   

2. RDN模型代码下载

   ```
   git clone https://github.com/yjn870/RDN-pytorch
   ```

   

3. 编写pth2onnx脚本RDN_pth2onnx.py

   **说明：**

   > 由于onnx 11版本的算子导出的模型会包含暂不支持的DepthToSpace算子，因此pth2onnx中用onnx 9的版本导出onnx模型

   

4. 执行pth2onnx脚本，生成onnx模型文件

   ```py
   python3.7 RDN_pth2onnx.py --input-file=rdn_x2.pth --output-file=rdn_x2.onnx
   ```



#### 3.2 onnx转om模型

1. 设置环境变量

   ```
   source env.sh
   ```

2. 使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01

   ```
   atc --framework=5 --model=rdn_x2.onnx --output=rdn_x2_bs1 --input_format=NCHW --input_shape="image:1,3,114,114" --log=debug --soc_version=Ascend310
   ```



#### 3.3 om模型性能优化

直接使用atc工具转换会将Transpose算子翻译为TransposeD，而TransposeD在一些输入形状下会有很差的性能。因此首先需要将Transpose的输入形状加入atc转换的白名单中。

1. 获取om模型的profilling文件

   运行[profilling分析脚本](https://gitee.com/wangjiangben_hw/ascend-pytorch-crowdintelligence-doc/tree/master/Ascend-PyTorch%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%8C%87%E5%AF%BC/%E4%B8%93%E9%A2%98%E6%A1%88%E4%BE%8B/%E7%9B%B8%E5%85%B3%E5%B7%A5%E5%85%B7/run_profiling)生成profilling目录，其中其中op_statistic_0_1.csv文件统计了模型中每类算子总体耗时与百分比，op_summary_0_1.csv中包含了模型每个算子的aicore耗时

2. 分析profilling文件

   使用profiling工具分析，可以从输出的csv文件看到算子统计结果

   | Model Name | OP Type    | Core Type | Count | Total Time(us) | Min Time(us) | Avg Time(us) | Max Time(us) | Ratio(%) |
   | ---------- | ---------- | --------- | ----- | -------------- | ------------ | ------------ | ------------ | -------- |
   | rdn_x2_bs1 | TransData  | AI Core   | 4     | 935253.6       | 63.80209     | 233813.4     | 934225.7     | 44.2515  |
   | rdn_x2_bs1 | TransposeD | AI Core   | 1     | 934287.8       | 934287.8     | 934287.8     | 934287.8     | 44.2058  |
   | rdn_x2_bs1 | Conv2D     | AI Core   | 150   | 126733.2       | 78.2812      | 844.8882     | 1563.906     | 5.996379 |
   | rdn_x2_bs1 | ConcatD    | AI Core   | 129   | 117099.7       | 193.9584     | 907.7495     | 1612.917     | 5.540568 |
   | rdn_x2_bs1 | Cast       | AI Core   | 2     | 121.6667       | 59.4792      | 60.83335     | 62.1875      | 0.005757 |

   可以看到TransposeD和TransData算子耗时很长，重点对TransposeD算子进行优化。

3. TransposeD算子优化

   将TransposeD算子的输入shape添加到白名单中，/usr/local/Ascend/ascend-toolkit/5.0.2/x86_64-linux/opp/op_impl/built-in/ai_core/tbe/impl/dynamic/transpose.py里添加shape白名单：[1, 64, 2, 2, 114, 114]

   ```python
   white_list_shape = [
       ...
       [1, 64, 2, 2, 114, 114]
   ]
   ```

4. 优化前后性能对比

   |        | TransposeD Avg Time(us) | 310 推理性能(FPS) |
   | :----: | :---------------------: | :---------------: |
   | 优化前 |       934287.8125       |       3.76        |
   | 优化后 |        10141.094        |       29.56       |

   优化TransposeD算子后，310性能超过T4基准性能，满足验收要求



### 4 数据集预处理

#### 4.1 数据集获取

RDN模型使用[Set5验证集](https://github.com/hengchuan/RDN-TensorFlow/tree/master/Test/Set5)的5张图片进行测试，图片存放在/root/dataset/set5下面。



#### 4.2 数据集预处理

使用 RDN_preprocess.py 脚本进行数据预处理，脚本执行命令：

```python
python3.7 RDN_preprocess.py --src-path=/root/dataset/set5 --save-path=./prep_dataset
```

预处理脚本会在./prep_dataset/label/下保存中心裁剪大小为228x228的预处理图片用于后处理验证精度，并对上述图片进行下采样处理之后将大小为114x114的bin文件保存至./prep_dataset/data/下面作为模型输入



#### 4.3 生成数据集信息文件

1. 生成数据集信息文件脚本gen_dataset_info.py

2. 执行生成数据集信息脚本，生成数据集信息文件

   ```python
   python3.7 gen_dataset_info.py bin ./prep_dataset/data ./RDN_prep_bin.info 114 114
   ```

   第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息



### 5 离线推理

####  5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN V100R020C10 推理benchmark工具用户指南 01



####  5.2 离线推理

```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./rdn_x2_bs1.om -input_text_path=./RDN_prep_bin.info -input_width=114 -input_height=114 -output_binary=False -useDvpp=False
```

输出结果默认保存在当前目录result/dumpOutput_device{0}，对应了Set5中每张图片的输出。



### 6 精度对比

#### 6.1 离线推理精度

调用RDN_postprocess.py来进行后处理

```python
python3.7 RDN_postprocess.py --pred-path=./result/dumpOutput_device0 --label-path=./prep_dataset/label --result-path=./result.json --width=114 --height=114 --scale=2
```

详细的结果输出在result.json文件中，并在控制台上输出评估的平均PSNR结果，result.json文件内容如下：

```json
{
    "title": "Overall statistical evaluation",
    "value": [
        {
            "key": "Number of images",
            "value": "5"
        },
        {
            "key": "Top1 PSNR",
            "value": "43.16"
        },
        {
            "key": "Top2 PSNR",
            "value": "38.9"
        },
        {
            "key": "Top3 PSNR",
            "value": "37.56"
        },
        {
            "key": "Top4 PSNR",
            "value": "36.8"
        },
        {
            "key": "Top5 PSNR",
            "value": "34.94"
        },
        {
            "key": "Avg PSNR",
            "value": "38.27"
        }
    ]
}
```



#### 6.2 精度对比

github仓库中给出的官方精度为PSNR：38.18，npu离线推理的精度为PSNR：38.27，故精度达标



### 7 性能对比

#### 7.1 npu性能数据

参照3.3所描述的方法，将Transpose的输入[1, 64, 2, 2, 114, 114]加入优化白名单。随后运行

```
source env.sh
atc --framework=5 --model=rdn_x2.onnx --output=rdn_x2_bs1 --input_format=NCHW --input_shape="image:1,3,114,114" --log=debug --soc_version=Ascend310
```

得到size为114的om模型



**benchmark工具在整个数据集上推理获得性能数据**

batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：

```
[e2e] throughputRate: 1.32351, latency: 3777.82
[data read] throughputRate: 263.518, moduleLatency: 3.7948
[preprocess] throughputRate: 3.93749, moduleLatency: 253.969
[infer] throughputRate: 6.8117, Interface throughputRate: 7.39423, moduleLatency: 139.064
[post] throughputRate: 3.24091, moduleLatency: 308.555
```

Interface throughputRate: 7.39423，7.39423x4=29.577既是batch1 310单卡吞吐率
bs1 310单卡吞吐率：7.39423x4=29.577fps/card



#### 7.2 T4性能数据

在装有T4卡的服务器上使用TensorRT测试gpu性能，测试过程请确保卡没有运行其他任务。

batch1性能：

```
trtexec --onnx=rdn_x2.onnx --fp16 --shapes=image:1x3x114x114
```

gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch

```
[09/04/2021-07:06:46] [I] GPU Compute
[09/04/2021-07:06:46] [I] min: 38.6577 ms
[09/04/2021-07:06:46] [I] max: 44.8702 ms
[09/04/2021-07:06:46] [I] mean: 39.3814 ms
[09/04/2021-07:06:46] [I] median: 39.0232 ms
[09/04/2021-07:06:46] [I] percentile: 44.8702 ms at 99%
[09/04/2021-07:06:46] [I] total compute time: 3.07175 s
```

batch1 t4单卡吞吐率：1000/(39.3814/1)=25.393fps



#### 7.3 性能对比

batch1：29.577fps > 25.393fps

310单个device的吞吐率乘4即单卡吞吐率比T4单卡的吞吐率大，故310性能高于T4性能，性能达标。