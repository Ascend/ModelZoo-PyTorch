# Segmenter模型推理指导

## 1. 模型概述

### 1.1 论文地址

[Segmenter: Transformer for Semantic Segmentation.](https://arxiv.org/pdf/2105.05633)

### 1.2 代码地址

开源仓:：[https://github.com/rstrudel/segmenter](https://github.com/rstrudel/segmenter)<br>
分支：master<br>
commit-id：20d1bfad354165ee45c3f65972a4d9c131f58d53<br>

说明：
 1. 推理模型为开源仓中的 Seg-L-Mask/16 模型；
 2. 使用 Cityscapes 的验证集进行精度验证。


## 2. 环境说明
该模型离线推理使用 Atlas 300I Pro 推理卡，所有步骤都在 [CANN 5.1.RC1](https://www.hiascend.com/software/cann/commercial) 环境下进行，CANN包以及相关驱动、固件的安装请参考 [软件安装](https://www.hiascend.com/document/detail/zh/canncommercial/51RC1/envdeployment/instg)。
### 2.1 安装依赖
```shell
conda create -n seg python=3.7.5
conda activate seg
pip install torch==1.9.0 torchvision
```

### 2.2 获取开源仓代码
```shell
git clone https://github.com/rstrudel/segmenter.git
cd segmenter
git checkout master
git reset --hard 20d1bfad354165ee45c3f65972a4d9c131f58d53
pip install -e .
```

## 3. 模型转换

### 3.1 下载模型权重与配置文件
```shell
# 下载模型权重
wget https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/cityscapes/seg_large_mask/checkpoint.pth
# 下载配置文件
wget https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/cityscapes/seg_large_mask/variant.yml
```
说明：配置文件 variant.yml 需要与模型权重 checkpoint.pth 放置于同一目录下。

### 3.2 Pytorch模型转ONNX模型

```shell
python3 segmenter_pytorch2onnx.py -c ${checkpoint-path} -o ${onnx-path}
```
参数说明：<br>
-c, --checkpoint-path: 权重文件路径<br>
-o, --onnx-path: 生成ONNX模型的保存路径<br>


### 3.3 ONNX模型转OM模型

1、设置环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
说明：该命令中使用CANN默认安装路径(/usr/local/Ascend/ascend-toolkit)中的环境变量，使用过程中请按照实际安装路径设置环境变量。

2、生成OM模型
ATC工具的使用请参考 [ATC模型转换](https://www.hiascend.com/document/detail/zh/canncommercial/51RC1/inferapplicationdev/atctool)
```shell
atc --framework=5 --model=${onnx-path} --output=${om-path} --input_format=NCHW --input_shape="input:${bs},3,768,768" --log=error --soc_version=Ascend${chip_name} --op_precision_mode=op_precision.ini
```
说明：<br>
--model 指定ONNX模型的路径<br>
--output 生成OM模型的保存路径<br>
执行命令前，需设置--input_shape参数中 bs的数值，例如：1、4、8、16 <br> 
chip_name 可通过`npu-smi info`命令查看，例：310P3 <br>
![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

## 4. 数据处理

### 4.1 下载数据集

下载标注数据：[gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1)<br>
下载原始图片：[leftImg8bit_travaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3)

本模型用 cityscapes 的 val 集来验证模型精度。其中 cityscapes/gtFine/val 目录下存放的是标注后的 groundtruth 文件，此模型只用到后缀为 ‘gtFine_labelTrainIds.png’ 的图片；cityscapes/leftImg8bit/val 目录下存放原始的测试图片，包含三个城市共500张街景图片。该模型需要的groudtruth和原始图片的目录结构如下：
```
cityscapes
|-- gtFine
|   |-- val
|       |-- frankfurt
|       |   |-- frankfurt_000000_000294_gtFine_labelTrainIds.png
|       |   |-- frankfurt_000000_000576_gtFine_labelTrainIds.png
|       |   |-- ......
|       |-- lindau
|       |   |-- lindau_000000_000019_gtFine_labelTrainIds.png
|       |   |-- lindau_000001_000019_gtFine_labelTrainIds.png
|       |   |-- ......
|       `-- munster
|           |-- munster_000000_000019_gtFine_labelTrainIds.png
|           |-- munster_000001_000019_gtFine_labelTrainIds.png
|           |-- ......
`-- leftImg8bit
    |-- val
        |-- frankfurt
        |   |-- frankfurt_000000_000294_leftImg8bit.png
        |   |-- frankfurt_000000_000576_leftImg8bit.png
        |   |-- ......
        |-- lindau
        |   |-- lindau_000000_000019_leftImg8bit.png
        |   |-- lindau_000001_000019_leftImg8bit.png
        |   |-- ......
        `-- munster
            |-- munster_000000_000019_leftImg8bit.png
            |-- munster_000001_000019_leftImg8bit.png
            |-- ......
```

### 4.2 数据集预处理
```shell
python3 segmenter_preprocess.py --cfg-path ${cfg-path} --data-root ${data-root} --bin-dir ${bin-dir} --gt-path ${gt-path}
```
参数说明：<br>
--cfg-path: 模型的配置文件<br>
--data-root: cityscapes数据集所在的父目录<br>
--bin-dir: 指定预处理结果的存放目录<br>
--gt-path: 存放原始图片到标注图片的映射<br>


## 5. 离线推理

### 5.1 准备推理工具
推理工具使用ais_infer，须自己拉取源码，打包并安装
```shell
# 指定CANN包的安装路径
export CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest
# 获取源码
git clone https://gitee.com/ascend/tools.git
cd tools/ais-bench_workload/tool/ais_infer/backend/
# 打包
pip3.7 wheel ./
# 安装
pip3 install --force-reinstall ./aclruntime-0.0.1-cp37-cp37m-linux_aarch64.whl
```
参考：[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer#%E4%BB%8B%E7%BB%8D)


### 5.2 离线推理
```shell
cd tools/ais-bench_workload/tool/ais_infer/
mkdir ${result-dir}   # 提前创建推理结果的保存目录
python3 ais_infer.py --model ${om-path} --input ${bin-dir} --output ${result-dir}
```
参数说明：<br>
--model:  OM模型路径<br>
--input: 预处理bin文件所在目录的路径<br>
--output: 推理结果的保存目录<br>

### 5.3 精度验证
根据 4.2 生成的groudtruth文件和 5.2 生成的推理结果，可计算出模型的精度。
```shell
python3 segmenter_postprocess.py --result-dir {infer-result-dir} --gt-path ${gt-path} --metrics-path ${metrics-txt-path}
```
参数说明：<br>
--result-dir 存放推理结果的目录路径<br>
--gt-path 预处理生成的图片到标注的映射文件路径<br>
--metrics-path 可选，指定一个路径用于记录模型指标<br>


### 5.4 性能验证
用ais_infer工具进行纯推理100次，然后根据平均耗时计算出吞吐率
```shell
cd tools/ais-bench_workload/tool/ais_infer/
mkdir tmp_out   # 提前创建临时目录用于存放推理的临时输出
python3 ais_infer.py --model ${om-path} --output tmp_out --loop 100
rm -r tmp_out   # 删除临时目录
```
说明：性能测试前使用`npu-smi info`命令查看 NPU 设备的状态，确认空闲后再进行测试。

执行上述脚本后，日志中 Performance Summary 一栏会记录性能相关的指标，找到 NPU_compute_time 中的 mean 字段，即为NPU计算的平均耗时(ms)。以此计算出模型在对应 batch_size 下的吞吐率：
$$ 吞吐率 = \frac {bs * 1000} {mean}$$


## 6. 指标对比
总结：
 1. 310P上离线推理的精度(78.89%)与Pytorch在线推理精度(79.1%)持平；
 2. 性能最优的batch_size为 1，310P性能 / 性能基准 = 1.11 倍，已通过性能评审。

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
	<td rowspan="4">Seg-L-Mask/16</td>
	<td>1</td>
	<td rowspan="4">mIoU (SS) :<br>79.1%</td>
	<td rowspan="4">mIoU (SS) :<br>78.89%</td>
	<td>3.55 fps</td>
	<td>3.93 fps</td>
</tr>
<tr>
	<td>4</td>
	<td>2.69 fps</td>
	<td>3.42 fps</td>
</tr>
<tr>
	<td>8</td>
	<td>2.62 fps</td>
	<td>3.30 fps</td>
</tr>
<tr>
	<td>16</td>
	<td>2.38 fps</td>
	<td>3.30 fps</td>
</tr>
</table>
