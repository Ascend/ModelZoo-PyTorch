## RDN Onnx模型PyTorch端到端推理指导

### 1 模型概述

#### 1.1 论文地址

[RDN论文](https://arxiv.org/abs/1802.08797)



#### 1.2 代码地址

[RDN代码](https://github.com/yjn870/RDN-pytorch)



### 2 环境说明

#### 2.1 深度学习框架

```
CANN 5.1
pytorch = 1.5.0
torchvision = 0.6.0
onnx = 1.7.0
```



#### 2.2 python第三方库

```
numpy == 1.21.2
Pillow == 9.1.0
opencv-python == 4.5.5.64
mmcv == 1.5.1
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

   

4. 执行pth2onnx脚本，生成onnx模型文件

   ```py
   python3.7 RDN_pth2onnx.py --input-file=rdn_x2.pth --output-file=rdn_x2.onnx
   ```



#### 3.2 onnx转om模型

1. 设置环境变量

   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

2. 使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01

    ${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

   ```
   atc --framework=5 --model=rdn_x2.onnx --output=rdn_x2_bs1 --input_format=NCHW --input_shape="image:1,3,114,114" --log=debug --soc_version=Ascend${chip_name}
   ```



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

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310P上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN V100R020C10 推理benchmark工具用户指南 01



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

github仓库中给出的官方精度为PSNR：38.18，310离线推理的精度为PSNR：38.27，310P离线推理的精度为PSNR：38.27，故精度达标



### 7 性能对比

#### 7.1 npu(310)性能数据

运行

```
atc --framework=5 --model=rdn_x2.onnx --output=rdn_x2_bs1 --input_format=NCHW --input_shape="image:1,3,114,114" --log=debug --soc_version=Ascend310
```

得到size为114的om模型



**benchmark工具在整个数据集上推理获得性能数据**

batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：

```
[e2e] throughputRate: 3.60508, latency: 1386.93
[data read] throughputRate: 5656.11, moduleLatency: 0.1768
[preprocess] throughputRate: 25.072, moduleLatency: 39.8852
[inference] throughputRate: 7.36218, Interface throughputRate: 7.41332, moduleLatency: 135.745
[postprocess] throughputRate: 7.93869, moduleLatency: 125.965
```

Interface throughputRate: 7.41332，7.41332x4=29.653即是batch1 310单卡吞吐率
bs1 310单卡吞吐率：7.41332x4=29.653fps/card



#### 7.2 npu(310P)性能数据

运行

```
atc --framework=5 --model=rdn_x2.onnx --output=rdn_x2_bs1 --input_format=NCHW --input_shape="image:1,3,114,114" --log=debug --soc_version=Ascend${chip_name}
```

得到size为114的om模型



**benchmark工具在整个数据集上推理获得性能数据**

batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：

```
[e2e] throughputRate: 2.63637, latency: 1896.55
[data read] throughputRate: 571.102, moduleLatency: 1.751
[preprocess] throughputRate: 8.56546, moduleLatency: 116.748
[inference] throughputRate: 30.2376, Interface throughputRate: 47.3451, moduleLatency: 30.1382
[postprocess] throughputRate: 10.5238, moduleLatency: 95.0225
```

Interface throughputRate: 47.3451
bs1 310P单卡吞吐率：47.345fps/card



#### 7.3 T4性能数据

在装有T4卡的服务器上使用TensorRT测试gpu性能，测试过程请确保卡没有运行其他任务。

batch1性能：

```
trtexec --onnx=rdn_x2.onnx --fp16 --shapes=image:1x3x114x114
```

gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch

```
[05/10/2022-20:47:53] [I] GPU Compute
[05/10/2022-20:47:53] [I] min: 39.166 ms
[05/10/2022-20:47:53] [I] max: 48.2043 ms
[05/10/2022-20:47:53] [I] mean: 44.9095 ms
[05/10/2022-20:47:53] [I] median: 44.9874 ms
[05/10/2022-20:47:53] [I] percentile: 48.2043 ms at 99%
[05/10/2022-20:47:53] [I] total compute time: 3.09876 s 
```

batch1 t4单卡吞吐率：1000/(44.9095/1)=22.267fps



#### 7.3 性能对比

batch1：
310P/310 = 47.345fps/29.653fps = 1.597 > 1.2
310P/T4 = 47.345fps/22.267fps = 2.126 > 1.6

310P单卡吞吐率大于1.2倍310单卡吞吐率，且大于1.6倍T4单卡吞吐率，故性能达标。