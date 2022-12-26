# ShiftViT模型离线推理指导

## 1. 模型概述

### 1.1 论文地址

ShiftViT ([When Shift Operation Meets Vision Transformer: An Extremely Simple Alternative to Attention Mechanism](https://arxiv.org/abs/2201.10801))

### 1.2 代码地址

开源仓：[https://github.com/microsoft/SPACH](https://github.com/microsoft/SPACH)<br>
branch：main<br>
commit-id：20d1bfad354165ee45c3f65972a4d9c131f58d53<br>

说明：
1. 本项目中的推理模型为开源仓中的 **Shift-T/light** 模型；
2. 使用 ImageNet-1K 的验证集进行精度验证。


## 2. 搭建环境
该模型离线推理使用 Atlas 300I Pro 推理卡，所有步骤都在 [CANN 5.1.RC2](https://www.hiascend.com/software/cann/commercial) 环境下进行，CANN 包以及对应驱动、固件的安装请参考 [软件安装](https://www.hiascend.com/document/detail/zh/canncommercial/51RC2/envdeployment/instg)。
### 2.1 安装依赖
```shell
conda create -n spach python=3.7.5
conda activate spach
pip install torch==1.9.0
pip install torchvision==0.10.0
pip install timm==0.3.2
pip install einops==0.3.2
pip install onnx==1.12.0
pip install tqdm==4.64.0
pip install decorator==5.1.1
```

### 2.2 获取开源仓代码
```shell
git clone https://github.com/microsoft/SPACH.git
cd SPACH
git checkout main
git reset --hard f69157d4e90fff88285766a4eabf51b29d772da3
```
说明： 本仓中的脚本会引用开源仓代码，建议在开源仓目录下运行本仓脚本。

## 3. 模型转换

### 3.1 准备工作
```shell
# 下载 Pytorch 模型权重
mkdir shiftvit
wget https://github.com/microsoft/SPACH/releases/download/v1.0/shiftvit_tiny_light.pth -P shiftvit/

# timm 包内的 models/layers/helpers.py 与 torch 1.10.0 存在冲突，需修改
patch -p0 path/to/envs/spach/lib/python3.7/site-packages/timm/models/layers/helpers.py helpers.patch
```

### 3.2 Pytorch模型转ONNX模型


- step1 为提升模型性能，对模型源码 shift_feat 方法中 feature 数值更新操作进行等价替换
    ```shell
    patch -p0 models/shiftvit.py shiftvit.patch
    ```

- step2 生成ONNX模型
    ```shell
    python shiftvit_pytorch2onnx.py -c shiftvit/shiftvit_tiny_light.pth -o shiftvit/shiftvit1.onnx
    ```
    参数说明：
    - -c, --checkpoint-path,  PyTorch 权重文件路径
    - -o, --onnx-path, ONNX 模型的保存路径
    - -v, --opset-version, ONNX 算子集版本, 默认 12

- step3 为提升模型精度，对 ONNX 模型进行修改
    ```shell
    python modify_onnx.py -i shiftvit/shiftvit1.onnx -o shiftvit/shiftvit2.onnx
    ```
    参数说明：
    - -i, --input-onnx-path, 原始 ONNX 模型的路径
    - -o, --output-onnx-path, 修改后 ONNX 模型的保存路径

### 3.3 ONNX模型转OM模型

- step1 设置环境变量
    ```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```
    说明：该命令中使用 CANN 默认安装路径(/usr/local/Ascend/ascend-toolkit)中的环境变量，使用过程中请按照实际安装路径设置环境变量。

- step2 生成OM模型
    ATC 工具的使用请参考 [ATC模型转换](https://www.hiascend.com/document/detail/zh/canncommercial/51RC2/inferapplicationdev/atctool)
    ```shell
    atc --framework=5 \
        --model=shiftvit/shiftvit2.onnx \
        --output=shiftvit/shiftvit2-bs${bs} \
        --input_format=NCHW \
        --input_shape="input:${bs},3,224,224" \
        --log=error \
        --soc_version=Ascend${chip_name}
    ```
    参数说明：
    - --model, 指定ONNX模型的路径
    - --output, 生成OM模型的保存路径
    - --input_shape, 模型输入的 name 与 shape, 执行命令前，需设置 bs 的数值，例如：1、4、8、16、32、64
    - --soc_version, 昇腾AI处理器的版本，chip_name 可通过`npu-smi info`命令查看，例：310P3 <br>
        ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

## 4. 数据处理

### 4.1 准备原始数据集

本离线推理项目使用 ILSVRC2012 数据集（ImageNet-1k）的验证集进行精度验证。从 http://image-net.org/ 下载数据集并解压，
其中 val 的目录结构遵循 [torchvision.datasets.ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html#torchvision.datasets.ImageFolder) 的标准格式：
```
/path/to/imagenet/
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### 4.2 数据集预处理
```shell
python shiftvit_preprocess.py --data-root /path/to/imagenet/ --save-dir shiftvit/val-bin --gt-path shiftvit/val-gt.npy
```
参数说明：
- --batch-size, 每个 bin文件包含多少张原始图片数据，默认为 1
- --data-root, imagenet 所在路径
- --save-dir, 存放生成 bin 文件的目录路径
- --gt-path, groundtruth 路径，存放图片的分类标签


## 5. 离线推理

### 5.1 准备推理工具
推理工具使用ais_bench，须自己拉取源码，打包并安装
```shell
# 指定CANN包的安装路径
export CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest
```
请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。  


### 5.2 离线推理

- step1 准备工作

    ```shell
    # 设置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    # 提前创建推理结果的保存目录
    mkdir path/to/shiftvit/val-out/
    ```

- step2 对预处理后的数据进行推理

    ```shell
    python3 -m ais_bench --model /path/to/shiftvit/shiftvit2-bs${bs}.om --input path/to/shiftvit/val-bin --output path/to/shiftvit/val-out/
    ```
    参数说明:
    - --model, OM模型路径
    - --input, 存放预处理 bin 文件的目录路径
    - --output, 保存推理结果的目录路径

### 5.3 精度验证
根据 4.2 生成的 groudtruth 文件和 5.2 生成的推理结果，可计算出 OM 模型的离线推理精度。
```shell
python shiftvit_postprocess.py --result-dir shiftvit/val-out/2022_08_09-20_37_12/ --gt-path shiftvit/val-gt.npy
```
参数说明：
- --result-dir, 存放推理结果的目录路径，需注意ais_bench工具会在推理时根据推理开始时间创建子目录
- --gt-path, 预处理生成的图片到标注的映射文件路径


### 5.4 性能验证
用 ais_bench 工具进行纯推理100次，然后根据平均耗时计算出吞吐率
```shell
cd tools/ais-bench_workload/tool/ais_infer/
mkdir tmp_out   # 提前创建临时目录用于存放纯推理输出
python3 -m ais_bench --model /path/to/shiftvit/shiftvit2-bs${bs}.om --output tmp_out --batchsize ${bs} --loop 100
rm -r tmp_out   # 删除临时目录
```
说明：
1. **性能测试前使用`npu-smi info`命令查看 NPU 设备的状态，确认空闲后再进行测试。否则测出来性能会低于模型真实性能。**
2. 执行上述脚本后，日志中 **Performance Summary** 一栏会记录性能相关的指标，找到以关键字 **throughput** 开头的一行，即为 OM 模型的吞吐率。


## 6. 指标对比
总结：
 1. 310P上离线推理的精度(79.3%)与 PyTorch 在线推理精度(79.4%)持平；
 2. 性能最优的 batch_size 为 8，310P性能 / 性能基准 = 5.51 倍。

指标详情如下：
<table>
<tr>
	<th>模型</th>
	<th>batch_size</th>
	<th>Pytorch精度</th>
	<th>310P精度</th>
	<th>性能基准</th>
	<th>310P性能</th>
</tr>
<tr>
	<td rowspan="6">Shift-T/light</td>
	<td>1</td>
	<td rowspan="6">Acc@Top1:<br>79.4%</td>
	<td rowspan="6">Acc@Top1:<br>79.3%</td>
	<td>81.36 fps</td>
	<td>343.75 fps</td>
</tr>
<tr>
	<td>4</td>
	<td>134.40 fps</td>
	<td>730.24 fps</td>
</tr>
<tr>
	<td>8</td>
	<td>144.05 fps</td>
	<td>862.72 fps</td>
</tr>
<tr>
	<td>16</td>
	<td>149.74 fps</td>
	<td>855.70 fps</td>
</tr>
<tr>
	<td>32</td>
	<td>156.20 fps</td>
	<td>834.03 fps</td>
</tr>
<tr>
	<td>64</td>
	<td>156.48 fps</td>
	<td>842.44 fps</td>
</tr>
</table>

