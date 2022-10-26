# STDC ONNX模型端到端推理指导
- [1. 模型概述](#1)
    - [论文地址](#11)
    - [开源代码仓地址](#12)
- [2. 环境说明](#2)
    - [安装依赖](#21)
    - [获取开源模型代码仓](#22)
    - [准备数据集](#23)
- [3. 模型转换](#3)
    - [pth转onnx模型](#31)
    - [onnx转om模型](#32)
- [4. 数据预处理](#4)
- [5. 离线推理](#5)
    - [离线推理](#52)
    - [精度和性能比较](#53)

## <a name="1">1. 模型概述</a>
### <a name="11">1.1 论文地址</a>
[STDC论文](https://arxiv.org/pdf/2104.13188.pdf)
### <a name="12">1.2 开源代码仓地址</a>
[STDC模型开源代码仓](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/stdc)

> **说明：**   
> 本离线推理项目中STDC模型对应开源代码仓中STDC1 (No Pretrain)，以下说明中将STDC1 (No Pretrain)简称为STDC

## <a name="2">2. 环境说明</a>
### <a name="21">2.1 安装依赖</a>

```
pip install -r requirements.txt
```
CANN版本
```
CANN 5.1.RC1
```

### <a name="22">2.2 获取开源模型代码仓</a>
```
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .
```

### <a name="23">2.3 准备数据集</a>
注册Cityscapes后下载[gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1)和[leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3), 放在mmsegmentation目录下
```
cd mmsegmentation
mkdir data
cd data
ln -s $DATA_ROOT cityscapes
```
> **注：**   $DATA_ROOT用数据集地址代替

## <a name="3">3. 模型转换</a>

### <a name="31">3.1 pth转onnx模型</a>
1. 一键获取pth、onnx、om脚本命令：
```
bash test/pth2om.sh {chip_name}
```
{chip_name}为芯片型号，可通过"npu-smi info"命令查看。
指行后在mmsegmentation目录下获得权重文件：stdc1_512x1024_80k_cityscapes_20220224_073048-74e6920a.pth。  
在当前目录获得onnx模型、优化后onnx模型和om模型：stdc_bs1.onnx，stdc_optimize_bs1.onnx，stdc_optimize_bs1.om。  

## <a name="4">4. 数据预处理</a>
1. 执行数据预处理脚本
```
python ./STDC_preprocess.py /opt/npu/cityscapes/leftImg8bit/val/ ./prep_dataset
```
处理后的 bin 文件放在 prep_dataset 目录下。  

## <a name="5">5. 离线推理</a>

1. 准备 msame 推理工具
将 [msame](https://gitee.com/ascend/tools/tree/master/msame) 安装到当前工作目录
&nbsp;
2. 一键执行推理
```
bash ./test/eval_acc_perf.sh {dataset_root}
```
{dataset_root}为用户环境中cityscapes数集所在目录。
执行后prep_dataset保存的为预处理数据集文件，output为离线推理结果，postprocess_result.txt推理结果评价指标。  

**精度评测结果：**

| 模型    | 参考精度 | 310P 精度 | 基准性能 | 310P 性能 |
| ------- | ------- | -------- | -------- | -------- |
| STDC1 (No Pretrain) bs1  | mIoU = 71.82 | mIoU = 71.81 | 39.54 fps | 57.54 fps |