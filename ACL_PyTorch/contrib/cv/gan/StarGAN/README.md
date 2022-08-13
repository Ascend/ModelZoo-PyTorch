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
- 310:
```py
bash ./test/pth2om.sh './models/200000-G.pth'
```

- 710:
```py
bash ./test_710/pth2om.sh './models/200000-G.pth'
```


### 4 生成输入数据并保存为.bin文件

- 数据集默认路径为 `./data/celeba.zip` ，使用脚本 `unzip_dataset.sh` 解压数据集。

```
bash unzip_dataset.sh
```

- 使用脚本 `StarGAN_pre_processing.py` 获得二进制 bin 文件和基准的图片结果。

```
source ./test/env_npu.sh
python3.7 StarGAN_pre_processing.py --mode test  --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
                  --model_save_dir './models' --result_dir './result_baseline' \
                  --attr_path './data/celeba/list_attr_celeba.txt' --celeba_image_dir './data/celeba/images'
```



### 5 离线推理

####  5.1 msame工具概述

msame 工具为华为自研的模型推理工具，输入 om 模型和模型所需要的输入 bin 文件，输出模型的输出数据文件。模型必须是通过 atc 工具转换的 om 模型，输入 bin 文件需要符合模型的输入要求，且支持模型多输入。
```
git clone https://gitee.com/ascend/tools.git

export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub

cd tools/msame
dos2unix *.sh
chmod u+x  build.sh
./build.sh g++  ./msame/out   
cp ./out/main   ../../   #本步骤要在“/tools/msame”文件夹目录下执行
cd ../..

```

```
chmod 777 main
mv main msame
```

####  5.2 离线推理

```
bash ./test/eval_bs1_perf.sh
bash ./test/eval_bs4_perf.sh
bash ./test/eval_bs8_perf.sh
bash ./test/eval_bs16_perf.sh
```

- 使用脚本 `StarGAN_pre_processing_32_64.py` 获得二进制 bin 文件和基准的图片结果。
```
source ./test/env_npu.sh
python3.7 StarGAN_pre_processing_32_64.py --mode test  --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
                  --model_save_dir './models' --result_dir './result_baseline' \
                  --attr_path './data/celeba/list_attr_celeba.txt' --celeba_image_dir './data/celeba/images'
```
```
rm ./bin/img/155.bin
rm ./bin/attr/155.bin
rm ./bin/img/156.bin
rm ./bin/attr/156.bin
rm ./bin/img/157.bin
rm ./bin/attr/157.bin
rm ./bin/img/158.bin
rm ./bin/attr/158.bin
rm ./bin/img/159.bin
rm ./bin/attr/159.bin

bash ./test/eval_bs32_perf.sh
bash ./test/eval_bs64_perf.sh
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
(310 bs1) Inference average time: 318.90 ms
(310 bs1) FPS:200.690
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

#### 7.3 NPU 710 性能数据
```
(710 bs1) Inference average time: 1.32 ms
(710 bs1) FPS:757.576
```

根据时延和核心数，计算得到 Batchsize = 1 时单卡吞吐率 757.576 FPS

```
(710 bs16) Inference average time: 27.00 ms
(710 bs16) FPS:592.593
```

根据时延和核心数，计算得到 Batchsize = 16 时单卡吞吐率 592.593 FPS

#### 7.4 性能对比

| Batch Size | 310 (FPS/Card) | 710 (FPS/Card)| T4 (FPS/Card) | 310/T4   | 710/310  |  710/T4  | 
| ---------- | -------------- | ------------- | ------------- | -------- | -------- | -------- |
| 1          | *189.753*      | *757.576*     | *187.135*     | *101.4%* | *396.8%* | *375.0%* |
| 4          | *201.207*      | *923.788*     | *203.666*     | *98.80%* | *458.3%* | *451.9%* |
| 8          | *199.913*      | *984.010*     | *219.700*     | *91.00%* | *491.3%* | *457.2%* |
| 16         | *200.986*      | *592.593*     | *235.980*     | *85.17%* | *295.2%* | *261.3%* |
| 32         | *200.986*      | *991.633*     | *202.280*     | *99.36%* | *493.3%* | *490.2%* |
| 64         | *201.307*      | *1040.31*     | *195.670*     | *102.8%* | *516.7%* | *531.6%* |


