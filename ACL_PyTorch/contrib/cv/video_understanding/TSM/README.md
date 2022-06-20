# 基于开源mmaction2预训练的TSM模型端到端推理指导
-   [1 模型概述](#1-模型概述)
	-   [1.1 论文地址](#11-论文地址)
	-   [1.2 代码地址](#12-代码地址)
-   [2 环境说明](#2-环境说明)
	-   [2.1 深度学习框架](#21-深度学习框架)
	-   [2.2 python第三方库](#22-python第三方库)
-   [3 模型转换](#3-模型转换)
	-   [3.1 pth转onnx模型](#31-pth转onnx模型)
	-   [3.2 onnx转om模型](#32-onnx转om模型)
-   [4 数据集预处理](#4-数据集预处理)
	-   [4.1 数据集获取](#41-数据集获取)
	-   [4.2 数据集预处理](#42-数据集预处理)
	-   [4.3 生成数据集信息文件](#43-生成数据集信息文件)
-   [5 离线推理](#5-离线推理)
	-   [5.1 benchmark工具概述](#51-benchmark工具概述)
	-   [5.2 离线推理](#52-离线推理)
-   [6 精度对比](#6-精度对比)
	-   [6.1 离线推理精度统计](#61-离线推理精度统计)
	-   [6.2 开源精度](#62-开源精度)
	-   [6.3 精度对比](#63-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)
	-   [7.2 T4性能数据](#72-T4性能数据)
	-   [7.3 性能对比](#73-性能对比)

## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[TSM论文](https://arxiv.org/abs/1811.08383)  
TSM是一种通用且有效的时间偏移模块，它具有高效率和高性能，可以在达到3D CNN性能的同时，保持2D CNN的复杂性。TSM沿时间维度移动部分通道，从而促进相邻帧之间的信息交换。TSM可以插入到2D CNN中以实现零计算和零参数的时间建模。TSM可以扩展到在线设置，从而实现实时低延迟在线视频识别和视频对象检测。

### 1.2 代码地址
[mmaction2框架TSM代码](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm)

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
onnx==1.9.0
torch==1.9.0
torchvision==0.10.0
```

### 2.2 python第三方库
```
numpy==1.21.0
opencv-python==4.5.3.56
mmcv==1.3.9
```

**说明：** 
>   X86架构：opencv,pytorch,torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：opencv,pytorch,torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

atc暂不支持动态shape小算子，可以使用大颗粒算子替换这些小算子规避，这些小算子可以在转onnx时的verbose打印中找到其对应的python代码，从而根据功能用大颗粒算子替换，onnx能推导出变量正确的shape与算子属性正确即可，变量实际的数值无关紧要，因此这些大算子函数的功能实现无关紧要，因包含自定义算子需要去掉对onnx模型的校验。

### 3.1 pth转onnx模型
1.下载pth权重文件
[TSM基于mmaction2预训练的权重文件](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb_20210630-1fae312b.pth)
```
wget https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb_20210630-1fae312b.pth
```

2.mmaction2源码安装
```shell
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -r requirements/build.txt
pip install -v -e .
cd ..
```

**说明：**  
> 安装所需的依赖说明请参考mmaction2/docs/install.md

3.转换onnx
```shell
python pytorch2onnx.py mmaction2/configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb.py ./tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb_20210630-1fae312b.pth --output-file=tsm.onnx --softmax --verify --show --shape 1 8 3 224 224
```

4.简化onnx
使用onnxsim去除atc工具不支持的pad动态shape
```shell
python3.7 -m onnxsim --input-shape="1,8,3,224,224" tsm.onnx onnx_sim/tsm_bs1.onnx
```
若要获得不同batch_size的简化模型，只需要修改--input-shape参数，例如batch_size=16
```shell
python3.7 -m onnxsim --input-shape="16,8,3,224,224" tsm.onnx onnx_sim/tsm_bs16.onnx
```

### 3.2 onnx转om模型

1.设置环境变量
```shell
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/
export REPEAT_TUNE=True
```
上述环境变量可通过运行脚本添加
```
source env.sh
```

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373),
需要指定输出节点以去除无用输出，节点序号可能会因网络结构不同而不同，使用netron开源可视化工具查看具体的输出节点名。我们使用auto_tune对于模型进行优化，且由于Transdata耗时较短，Transdata白名单无法起到显著效果。
```shell
atc --model=onnx_sim/tsm_bs1.onnx --framework=5 --output=om/tsm_bs1 --input_format=NCDHW --input_shape="video:1,8,3,224,224" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"
```
若需要获取不同batch_size输入的om模型，则可以通过设定input_shape进行指定。下面的命令可生成batch_size=16的模型。
```shell
atc --model=onnx_sim/tsm_bs16.onnx --framework=5 --output=om/tsm_bs16 --input_format=NCDHW --input_shape="video:16,8,3,224,224" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用 [UCF-101](https://www.crcv.ucf.edu/research/data-sets/ucf101/) 的验证集进行测试，数据集下载步骤如下
```shell
cd ./mmaction2/tools/data/ucf101
bash download_annotations.sh
bash download_videos.sh
bash extract_rgb_frames_opencv.sh
bash generate_videos_filelist.sh
bash generate_rawframes_filelist.sh
```
（可选）本项目默认将数据集存放于/opt/npu/
```
cd ..
mv /ucf101 /opt/npu/
```

### 4.2 数据集预处理
1.预处理脚本tsm_ucf101_preprocess.py
```python
import os
import argparse
from mmcv import Config
from mmaction.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset UCF101 Preprocessing')
    parser.add_argument('--config',
                        default='./mmaction2/configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb.py',
                        help='config file path')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for inference')
    parser.add_argument('--num_worker', default=8, type=int, help='Number of workers for inference')
    parser.add_argument('--data_root', default='./mmaction2/tools/data/ucf101/rawframes/', type=str)
    parser.add_argument('--ann_file', default='./mmaction2/tools/data/ucf101/ucf101_val_split_1_rawframes.txt', type=str)
    parser.add_argument('--name', default='out_bin', type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.data_prefix = args.data_root

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=args.batch_size,
        workers_per_gpu=args.num_worker,
        dist=False,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    root_path = os.path.dirname(args.ann_file)
    out_path = os.path.join(root_path, args.name)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    file = open(os.path.join(root_path, 'ucf101.info'), 'w')

    for i, data in enumerate(data_loader):
        print('Preprocessing video {}/{}'.format(i, len(data_loader)))
        imgs = data['imgs']
        label = data['label']

        for batch in range(imgs.shape[0]):
            l = label.cpu().numpy()[batch]
            file.write(str(args.batch_size*i+batch) + ' ' + str(l))
            file.write('\n')

        if imgs.shape[0] != args.batch_size:
            imgs = F.pad(imgs, (0,0,0,0,0,0,0,0,0,args.batch_size-imgs.shape[0]))

        bin = imgs.cpu().numpy()
        bin.tofile(out_path + '/' + str(i) + '.bin')


if __name__ == '__main__':
    main()
```

2.执行预处理脚本，生成数据集预处理后的bin文件以及相应的info文件
```shell
python tsm_ucf101_preprocess.py --batch_size 1 --data_root /opt/npu/ucf101/rawframes/ --ann_file /opt/npu/ucf101/ucf101_val_split_1_rawframes.txt --name out_bin_1
```
第一个参数为batch_size，第二个参数为图片所在路径，第三个参数为图片对应的信息（由bash generate_rawframes_filelist.sh生成），第四个参数为保存的bin文件和info文件所在目录的名称。

下面的命令可生成batch_size=16的数据文件以及相应的info文件
```shell
python tsm_ucf101_preprocess.py --batch_size 16 --data_root /opt/npu/ucf101/rawframes/ --ann_file /opt/npu/ucf101/ucf101_val_split_1_rawframes.txt --name out_bin_16
```

若数据集下载位置不同，请将数据集目录（/opt/npu/ucf101）替换为相应的目录，若按照上述步骤下载数据集至./mmaction2/tools/data/ucf101，则无需指定这两个目录参数。

预处理后的bin文件默认保存于{数据集目录}/{name}/，info文件保存为{数据集目录}/ucf101.info。

若需要测试不同batch_size下模型的性能，可以指定batch_size以及保存目录的名称name。

## 5 离线推理

-   **[使用msame工具推理](#51-使用msame工具推理)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 使用msame工具推理
1.首先需要获取msame工具
```shell
git clone https://gitee.com/ascend/tools.git
```

2.而后安装msame工具
```shell
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub
cd ./tools/msame
./build.sh g++ ./
cd ../..
```

3.增加执行权限
```shell
chmod u+x ./tools/msame/out/msame
```

### 5.2 离线推理
使用msame进行离线推理，下面的命令针对batch_size=1的情况进行推理
```shell
./tools/msame/out/msame --model ./om/tsm_bs1.om --input /opt/npu/ucf101/out_bin_1 --output ./output/out_bs1/ --outfmt TXT
```
第一个参数为om模型所在路径，第二个参数为上述预处理过程得到的bin文件，第三个参数为输出路径，第四个参数为输出格式，第五个参数为性能指标

输出结果默认保存在./out_bs1/{当前时间}/下，其内容为每一组数据的分类预测结果，对应为单个output_0.txt文件
```
  输出           shape        数据类型            数据含义
output_0        1*101          FP32           分类预测结果
```

执行如下命令可以针对batch_size=16的情况进行推理
```shell
./tools/msame/out/msame --model ./om/tsm_bs16.om --input /opt/npu/ucf101/out_bin_16 --output ./output/out_bs16/ --outfmt TXT
```

## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计
1.后处理统计精度脚本tsm_ucf101_postprocess.py
```python
import os
import argparse
import numpy as np
from collections import OrderedDict
from mmaction.core import top_k_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset UCF101 Postprocessing')
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--info_path', type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # load info file
    gt_labels = []
    with open(args.info_path, 'r') as f:
        for line in f.readlines():
            gt_labels.append(int(line.split(' ')[1]))

    # load inference result
    results = []
    num_file = len(os.listdir(args.result_path))
    for idx in range(num_file):
        file = os.path.join(args.result_path, str(idx) + '_output_0.txt')
        with open(file, 'r') as f:
            for batch in f.readlines():
                line = batch.split(' ')[:-1]
                line = np.array([float(x) for x in line])
                results.append(line)
    results = results[:len(gt_labels)]

    metrics = ['top_k_accuracy']
    metric_options = dict(top_k_accuracy=dict(topk=(1, 5)))
    eval_results = OrderedDict()
    for metric in metrics:
        print(f'Evaluating {metric} ...')
        if metric == 'top_k_accuracy':
            topk = metric_options.setdefault('top_k_accuracy', {}).setdefault('topk', (1, 5))
            if not isinstance(topk, (int, tuple)):
                raise TypeError(f'topk must be int or tuple of int, but got {type(topk)}')
            if isinstance(topk, int):
                topk = (topk,)

            top_k_acc = top_k_accuracy(results, gt_labels, topk)
            log_msg = []
            for k, acc in zip(topk, top_k_acc):
                eval_results[f'top{k}_acc'] = acc
                log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
            log_msg = ''.join(log_msg)
            print(log_msg)
            continue


if __name__ == '__main__':
    main()
```

2.执行后处理脚本，获取精度
```shell
python tsm_ucf101_postprocess.py --result_path ./output/out_bs1 --info_path /opt/npu/ucf101/ucf101.info
```
第一个参数为预测结果所在路径（需根据实际输出路径进行修改），第二个参数为数据集info文件路径

当batch_size=1时，执行完成后，程序会打印出精度：
```
Evaluating top_k_accuracy ...

top1_acc	0.9448
top5_acc	0.9963
```

下面的命令可以针对batch_size=16的情况计算精度
```shell
python tsm_ucf101_postprocess.py --result_path ./output/out_bs16/20210727_143344/ --info_path /opt/npu/ucf101/ucf101.info
```

### 6.2 开源精度
[官网精度](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb/20210605_182720.log)
```
{'top1_acc': 0.9450, 'top5_acc': 0.9958}
```

### 6.3 精度对比
不同batch_size下，om模型的离线推理精度如下
```
batch_size      top1_acc        top5_acc
    1            0.9448          0.9963
    4            0.9448          0.9963
    8            0.9448          0.9963
    16           0.9448          0.9963
    32           0.9448          0.9963
```
可以发现，不同batch_size下，om模型的推理性能并没有差异，且与开源精度相比，精度下降小于1%，精度达标

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  
-   **[T4性能数据](#72-T4性能数据)**  
-   **[性能对比](#73-性能对比)**  

### 7.1 npu性能数据
1. benchmark工具简述
benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，
可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)

2. 性能测试
测试npu性能要确保device空闲，使用npu-smi info命令可查看device是否在运行其它推理任务
```shell
./benchmark.x86_64 -round=20 -om_path=./om/tsm_bs1.om -device_id=0 -batch_size=1
```
执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果。通过设置不同的om模型以及batch_size，可以测试不同batch_size下om模型的性能

下面的命令可以测试batch_size=16的情况下的npu性能
```
./benchmark.x86_64 -round=20 -om_path=./om/tsm_bs16.om -device_id=0 -batch_size=16
```

batch_size=1时，310的单卡吞吐率：16.9917*4=67.967fps。输出结果如下
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_tsm_bs1_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 16.9917samples/s, ave_latency: 59.5304ms
```

batch_size=4时，310的单卡吞吐率：15.24*4=60.960fps。输出结果如下
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_tsm_bs4_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 15.24samples/s, ave_latency: 65.7176ms
```

batch_size=8时，310的单卡吞吐率：14.8116*4=59.246fps。输出结果如下
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_tsm_bs8_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 14.8116samples/s, ave_latency: 67.6128ms
```

batch_size=16时，310的单卡吞吐率：14.6556*4=58.622fps。输出结果如下
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_tsm_bs16_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 14.6556samples/s, ave_latency: 68.2717ms
```

batch_size=32时，310的单卡吞吐率：14.5255*4=58.102fps。输出结果如下
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_tsm_bs32_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 14.5255samples/s, ave_latency: 68.8549ms
```

### 7.2 T4性能数据
由于我们并没有使用自定义算子，因此可以直接使用TensorRT测试gpu性能。

测试batch_size=1的性能
```shell
trtexec --onnx=onnx_sim/tsm_bs1.onnx --fp16 --shapes=video:1x8x3x224x224 --threads
```

通过将参数shapes替换为其他batch_size，我们可以测试不同batch_size下的性能。例如batch_size=16时的命令如下
```
trtexec --onnx=onnx_sim/tsm_bs16.onnx --fp16 --shapes=video:16x8x3x224x224 --threads
```

batch_size=1时，T4的单卡吞吐率: 1000/(9.96763/1)=100.325fps
```shell
[08/02/2021-10:58:27] [I] GPU Compute
[08/02/2021-10:58:27] [I] min: 9.70959 ms
[08/02/2021-10:58:27] [I] max: 10.5096 ms
[08/02/2021-10:58:27] [I] mean: 9.96763 ms
[08/02/2021-10:58:27] [I] median: 9.93506 ms
[08/02/2021-10:58:27] [I] percentile: 10.4962 ms at 99%
[08/02/2021-10:58:27] [I] total compute time: 3.02019 s
```

batch_size=4时，T4的单卡吞吐率: 1000/(37.4986/4)=106.671fps
```shell
[08/02/2021-11:01:52] [I] GPU Compute
[08/02/2021-11:01:52] [I] min: 36.3376 ms
[08/02/2021-11:01:52] [I] max: 39.7435 ms
[08/02/2021-11:01:52] [I] mean: 37.4986 ms
[08/02/2021-11:01:52] [I] median: 37.2561 ms
[08/02/2021-11:01:52] [I] percentile: 39.7435 ms at 99%
[08/02/2021-11:01:52] [I] total compute time: 3.07489 s
```

batch_size=8时，T4的单卡吞吐率: 1000/(73.6846/8)=108.571fps
```shell
[08/02/2021-11:07:19] [I] GPU Compute
[08/02/2021-11:07:19] [I] min: 72.317 ms
[08/02/2021-11:07:19] [I] max: 78.6246 ms
[08/02/2021-11:07:19] [I] mean: 73.6846 ms
[08/02/2021-11:07:19] [I] median: 73.1267 ms
[08/02/2021-11:07:19] [I] percentile: 78.6246 ms at 99%
[08/02/2021-11:07:19] [I] total compute time: 3.09475 s
```

batch_size=16时，T4的单卡吞吐率: 1000/(144.696/16)=110.577fps
```shell
[08/02/2021-11:17:01] [I] GPU Compute
[08/02/2021-11:17:01] [I] min: 141.875 ms
[08/02/2021-11:17:01] [I] max: 152.407 ms
[08/02/2021-11:17:01] [I] mean: 144.696 ms
[08/02/2021-11:17:01] [I] median: 143.706 ms
[08/02/2021-11:17:01] [I] percentile: 152.407 ms at 99%
[08/02/2021-11:17:01] [I] total compute time: 3.18331 s
```

batch_size=32时，T4的单卡吞吐率: 1000/(288.588/32)=110.885fps
```shell
[08/02/2021-11:35:17] [I] GPU Compute
[08/02/2021-11:35:17] [I] min: 284.045 ms
[08/02/2021-11:35:17] [I] max: 292.737 ms
[08/02/2021-11:35:17] [I] mean: 288.588 ms
[08/02/2021-11:35:17] [I] median: 289.395 ms
[08/02/2021-11:35:17] [I] percentile: 292.737 ms at 99%
[08/02/2021-11:35:17] [I] total compute time: 3.46305 s
```

### 7.3 性能对比

Ascend310/GPU基准:

batch_size=1: 310/t4=67.967/100.325=0.68倍

batch_size=4: 310/t4=60.960/106.671=0.57倍

batch_size=8: 310/t4=59.246/108.571=0.55倍

batch_size=16: 310/t4=58.622/110.577=0.53倍

batch_size=32: 310/t4=58.102/110.885=0.52倍

由于模型并没有性能要求，bs1、bs4、bs8、bs16、bs32时npu的性能高于T4性能的0.5倍，性能达标

Ascend310P/GPU基准:

batch_size=1: 310P/t4=186/100=1.86倍

batch_size=4: 310P/t4=157/109=1.44倍

batch_size=8: 310P/t4=153/111=1.37倍

batch_size=16: 310P/t4=151/112=1.35倍

batch_size=32: 310P/t4=139/113=1.23倍

batch_size=64: 310P/t4=131/115=1.14倍

最优吞吐率对比：186/115=1.62倍
