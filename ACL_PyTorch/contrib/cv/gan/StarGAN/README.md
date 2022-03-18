## StarGAN Onnx 模型 PyTorch 端到端推理指导

### 1 模型概述

- 论文地址

```
https://arxiv.org/abs/1711.09020
```

- 代码地址

```
https://github.com/yunjey/stargan
```

- 数据集地址

```
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/dataset/celeba.zip
```



### 2 环境说明

```
CANN = 5.0.2
pytorch = 1.5.0
torchvision = 0.6.0
onnx = 1.8.0
numpy = 1.21.1
```

> X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
> Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装



### 3 pth 转 om 模型

- pth 权重文件默认路径为  `./models/200000-G.pth`
- 进入根目录 `./` 执行 `./test/pth2om` 脚本，自动生成生成 onnx 模型文件和om文件

```py
bash ./test/pth2om.sh './models/200000-G.pth'
```



### 4 生成输入数据并保存为.bin文件

- 数据集默认路径为 `./celeba.zip` ，使用脚本 `unzip_dataset.sh` 解压数据集。

```
bash unzip_dataset.sh
```

- 使用脚本 `StarGAN_pre_processing.py` 获得二进制 bin 文件和基准的图片结果。

```
source ./test/env_npu.sh
python3.7 StarGAN_pre_processing.py --mode test  --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
                  --model_save_dir './models' --result_dir './result_baseline' \
                  --attr_path './data/celeba/images' --celeba_image_dir './data/celeba/list_attr_celeba.txt'
```



### 5 离线推理

####  5.1 msame工具概述

msame 工具为华为自研的模型推理工具，输入 om 模型和模型所需要的输入 bin 文件，输出模型的输出数据文件。模型必须是通过 atc 工具转换的 om 模型，输入 bin 文件需要符合模型的输入要求，且支持模型多输入。

```
chmod 777 msame
```

####  5.2 离线推理

```
bash ./test/eval_bs1_perf.sh
bash ./test/eval_bs16_perf.sh
```

输出数据默认保存在根目录的 `./StarGAN_[yourBatchSize].log` 中，可以看到时延和 FPS。输出图片默认保存在当前目录 `output_[yourBatchSize]/` 下，为保存模型输入高维张量数据的 txt 文件。



### 6 精度对比

调用 ` StarGAN_post_processing.py` 来进行后处理，把输出的 txt 文件转换为输出图像。

```python
python3.7 StarGAN_post_processing.py --folder_path './output_bs1/[YYYYMMDD_HHMMSS]' --batch_size 1
python3.7 StarGAN_post_processing.py --folder_path './output_bs16/[YYYYMMDD_HHMMSS]' --batch_size 16
```

详细的结果输出在 `./output_[yourBatchSize]/jpg` 文件夹中，可以和 `result_baseline` 文件夹下的在线推理结果做对比。可以发现各个 batchsize 的离线推理生成的图片与基准基本一致。



### 7 性能对比

#### 7.1 NPU 310 性能数据
```
(310 bs1) Inference average time: 21.04 ms
(310 bs1) FPS:190.114
```

根据时延和核心数，计算得到 Batchsize = 1 时单卡吞吐率 190.114 FPS

```
(310 bs16) Inference average time: 313.39 ms
(310 bs16) FPS:204.218
```

根据时延和核心数，计算得到 Batchsize = 16 时单卡吞吐率 204.218 FPS

#### 7.2 GPU T4 性能数据

```
&&&& RUNNING TensorRT.trtexec # trtexec --onnx=StarGAN.onnx --shapes=real_img:1x3x128x128,attr:1x5
...
[11/10/2021-07:45:57] [I] GPU Compute
[11/10/2021-07:45:57] [I] min: 4.5766 ms
[11/10/2021-07:45:57] [I] max: 8.12921 ms
[11/10/2021-07:45:57] [I] mean: 5.34373 ms
[11/10/2021-07:45:57] [I] median: 5.32825 ms
[11/10/2021-07:45:57] [I] percentile: 6.91772 ms at 99%
[11/10/2021-07:45:57] [I] total compute time: 2.93371 s
```

根据时延和核心数，计算得到 Batchsize = 1 时单卡吞吐率 187.135 FPS

```
&&&& RUNNING TensorRT.trtexec # trtexec --onnx=StarGAN.onnx --shapes=real_img:16x3x128x128,attr:16x5
...
[11/10/2021-08:03:49] [I] GPU Compute
[11/10/2021-08:03:49] [I] min: 65.5917 ms
[11/10/2021-08:03:49] [I] max: 76.011 ms
[11/10/2021-08:03:49] [I] mean: 67.8021 ms
[11/10/2021-08:03:49] [I] median: 67.15 ms
[11/10/2021-08:03:49] [I] percentile: 76.011 ms at 99%
[11/10/2021-08:03:49] [I] total compute time: 3.1189 s
```

根据时延和核心数，计算得到 Batchsize = 16 时单卡吞吐率 235.980 FPS

#### 7.3 性能对比

| Batch Size | 310 (FPS/Card) | T4 (FPS/Card) | 310/T4   |
| ---------- | -------------- | ------------- | -------- |
| 1          | *189.753*      | *187.135*     | *101.4%* |
| 4          | *201.207*      | *203.666*     | *98.80%* |
| 8          | *199.913*      | *219.700*     | *91.00%* |
| 16         | *200.986*      | *235.980*     | *85.17%* |

