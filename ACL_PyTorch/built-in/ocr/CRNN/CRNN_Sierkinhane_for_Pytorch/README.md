# CRNN_Sierkinhane 推理指导

- [CRNN\_Sierkinhane 推理指导](#crnn_sierkinhane-推理指导)
  - [概述](#概述)
  - [详细步骤](#详细步骤)
    - [准备项目](#准备项目)
    - [准备数据集](#准备数据集)
    - [搭建环境](#搭建环境)
    - [转换模型](#转换模型)
    - [精度验证](#精度验证)
    - [纯推理性能验证](#纯推理性能验证)
  - [精度和性能](#精度和性能)
  - [动态推理的流程](#动态推理的流程)

## 概述

CRNN_Sierkinhane 是一个基于卷积循环网络的中文 OCR 模型。

参考实现：

```
https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec
branch=stable
commit_id=a565687c4076b729d4059593b7570dd388055af4
```

输入输出：

| 输入数据 | 数据类型 | 大小 | 数据排布格式 |
| ---- | ---- | ---- | ---- |
| input | FLOAT32  | <batch_size> x 1 x 32 x 160 | NCHW |

| 输出数据 | 数据类型 | 大小 | 数据排布格式 |
| ---- | ---- | ---- | ---- |
| output | FLOAT32 | 41 x <batch_size> x <字符集总数 + 1> | ND |

数据集：GitHub 仓库提供的 360 万数据集。

## 详细步骤

### 准备项目

1. 克隆 GitHub 仓库，切换到指定分支、指定 commit_id。

   ```bash
   git clone https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec.git
   cd CRNN_Chinese_Characters_Rec
   git checkout stable
   git reset --hard a565687c4076b729d4059593b7570dd388055af4
   ```

2. 在 CRNN_Chinese_Characters_Rec 目录下新建一个 npu_infer 文件夹，把 Gitee 仓库 CRNN_sierkinhane_for_Pytorch 目录下的所有文件复制到 npu_infer 文件夹。

### 准备数据集

1. 从 GitHub 仓库获取数据集，把图片数据放在 images/total_images 路径下。

   ```bash
   ll images/total_images | grep "^-" | wc -l # 输出训练集和测试集的图片总数量为 3644007
   wc -l lib/dataset/txt/test.txt # 输出测试集标签数量为 364400
   ```

2. 把 test.txt 文件中对应的图片复制出来。运行后生成 images/test_images.py 文件夹。

   ```bash
   python3 npu_infer/extract_test_images.py # 可以使用 -h 查看参数用法
   ll images/test_images | grep "^-" | wc -l # 输出测试集的图片数量为 364400
   ```

### 搭建环境

1. 环境版本。

   | 环境 | 版本 | 安装指导 |
   | ---- | ---- | ---- |
   | NPU 驱动和固件 | 23.0.rc1 | [安装指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha002/softwareinstall/instg/instg_000018.html) |
   | CANN | 6.2.T200 | [安装指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha002/softwareinstall/instg/instg_000036.html) |
   | Python | 3.7.5 | |
   | PyTorch | 1.13.1 | |

2. 新建 conda 环境，安装依赖。

   ```bash
   # conda 环境可选
   conda create -n crnn python=3.7.5
   conda activate crnn
   pip3 install -r my/requirements.txt
   ```

3. 设置环境变量。

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

### 转换模型

1. 导出 .onnx 文件。运行后生成 crnn.onnx 文件。

   ```bash
   python3 npu_infer/pth2onnx.py # 可以使用 -h 查看参数用法
   ```

2. 运行以下命令，从 Name 列查看芯片名称。

   ```bash
   npu-smi info
   ```

   比如，执行后可能显示如下信息，则芯片名称为 310P3。

   ```
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

3. 转换 .om 模型。运行后生成 om/crnn_bs${batch_size}.om 文件。

   ```bash
   atc \
       --framework=5 \
       --model=crnn.onnx \
       --output=om/crnn_bs${batch_size} \
       --input_format=NCHW \
       --input_shape="input:${batch_size},1,32,160" \
       --soc_version=Ascend${chip_name}
   ```

   备注：${chip_name} 请根据实际芯片名称填写。比如，芯片名称为 310P3，则传入参数 --soc_version=Ascend310P3。

   参数说明：

   - --framework：原始框架类型。
   - --model：原始模型文件路径与文件名。
   - --output：如果是开源框架的网络模型，存放转换后的离线模型的路径以及文件名。如果是单算子描述文件，存放转换后的单算子模型的路径。
   - --input_format：输入数据格式。
   - --input_shape：模型输入数据的shape。
   - --soc_version：模型转换时指定芯片版本。
   - 更多参数请参考 [atc 参数概览](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha002/infacldevg/atctool/atlasatc_16_0041.html)。

### 精度验证

1. 前处理。运行后生成 images/preprocessed_test_images 文件夹和 preprocess_test.txt 文件。

   ```bash
   python3 npu_infer/preprocess.py # 可以使用 -h 查看参数用法
   ```
   运行完会在`images/preprocessed_test_images`生成处理好的数据

2. 推理。运行后生成 ais_bench_output 文件夹。

   ```bash
   python3 -m ais_bench \
       --model=om/crnn_bs${batch_size}.om \
       --input=images/preprocessed_test_images \
       --output=ais_bench_output \
       --output_dirname=result \
       --output_batchsize_axis=1 \
       --outfmt=NPY
   ```

   参数说明：

   - --model：OM离线文件
   - --input：模型需要的输入
   - --output：推理结果保存目录
   - --output_dirname：推理结果保存子目录
   - --output_batchsize_axis：输出tensor的batchsize轴
   - --outfmt：输出数据的格式
   - 更多参数请参考 [ ais_bench 推理工具使用指南](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)。

3. 后处理。运行后在控制台输出精度。

   ```bash
   python3 npu_infer/postprocess.py # 可以使用 -h 查看参数用法
   ```

### 纯推理性能验证

```bash
python3 -m ais_bench --model=om/crnn_bs${batch_size}.om --loop=100
```
## 动态推理的流程

### 数据预处理
1. 由于验证集数量很大，为了加快测试流程，可以用以下命令复制前N张图片（这步可选）
```bash
cd images/test_images
mkdir ../tmp1000
ls | head -n 1000 | xargs -i cp {} ../tmp1000
cd -
```

2. 生成动态测试数据集
```bash
python3 npu_infer/preprocess.py --image-dir images/tmp1000 --is-dym True
```
运行完会在`images/preprocessed_tmp1000`生成处理好的数据

### 生成om模型
```bash
atc \
       --framework=5 \
       --model=crnn.onnx \
       --output=om/crnn_dym \
       --input_format=ND \
       --input_shape="input:1~64,1,32,32~2048" \
       --soc_version=Ascend${chip_name}
``` 
模型生成在`om/crnn_dym_linux_${arch}.om`，`${arch}`是服务器的对应架构

### 推理验证
```bash
python3 npu_infer/npu_end2end.py --data_path images/tmp1000 --om_path om/crnn_dym_linux_${arch}.om --output npu_result
```
运行完结果会保存在当前目录`npu_result`

### 后处理精度验证
```bash
python3 npu_infer/postprocess.py --predict-dir npu_result --is-dym True
```
运行完精度会打屏显示

## GPU动态Shape测试流程
1. 环境安装
在`ocr/utils`下有GPU动态Shape的样例，按照[readme](../../utils/trtexec_dynamic/README.md)搭建好环境
2. 运行脚本
```bash
python3 npu_infer/gpu_end2end.py --data_path images/tmp1000 --onnx_path crnn.onnx --output gpu_result
```
运行完结果会保存在当前目录`gpu_result`
3. 后处理精度验证
```bash
python3 npu_infer/postprocess.py --predict-dir gpu_result --is-dym True
```
运行完精度会打屏显示

## 精度和性能

1. 静态数据

| 芯片型号 | Batch Size | 数据集 | 精度 | 性能 |
| ---- | ---- | ---- | ---- | ----|
| 310P3 | 1 | GitHub 仓库提供的 360 万数据集 | 78.37% | 696 |
| 310P3 | 4 | GitHub 仓库提供的 360 万数据集 | 78.37% | 2598 |
| 310P3 | 8 | GitHub 仓库提供的 360 万数据集 | 78.37% | 3990 |
| 310P3 | 16 | GitHub 仓库提供的 360 万数据集 | 78.37% | 5754 |
| 310P3 | 32 | GitHub 仓库提供的 360 万数据集 | 78.37% | 7445 |
| 310P3 | 64 | GitHub 仓库提供的 360 万数据集 | 78.37% | 8066 |

2. 动态数据

| 芯片型号 |  数据集 | 精度 | 性能 |
| ---- | ---- | ---- | ----|
| 310P3 | GitHub 仓库提供的 360 万数据集 | 75.5% | 163 |