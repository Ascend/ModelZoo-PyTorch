# M2Det模型PyTorch离线推理指导

## 1 环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

```csharp
pip3.7 install -r requirements.txt  
```

2.获取，修改与安装开源模型代码
```
/
├── M2Det
├── test
|    ├── eval_acc_perf.sh
|    ├── perf_g.sh
|    └── pth2om.sh
├── env.sh
├── gen_dataset_info.py
├── LICENSE
├── M2Det.patch
├── M2Det_postprocess.py
├── M2Det_preprocess.py
├── M2Det_pth2onnx.py
├── README.md
└── requirements.txt
```

```csharp
git clone https://github.com/qijiezhao/M2Det.git
cd M2Det
git reset --hard de4a6241bf22f7e7f46cb5cb1eb95615fd0a5e12
patch -p1 < ../M2Det.patch
sh make.sh
mkdir weights
mkdir logs
mkdir eval
cd ..
mkdir result
```

3.获取权重文件

权重文件放在M2Det/weights目录下

[m2det512_vgg.pth](https://pan.baidu.com/s/1LDkpsQfpaGq_LECQItxRFQ)

[vgg16_reducedfc.pth](https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth)

4.数据集

获取[val2014](http://images.cocodataset.org/zips/val2014.zip)、[instances_minival2014.json](http://datasets.d2.mpi-inf.mpg.de/hosang17cvpr/coco_minival2014.tar.gz)

创建目录

```
annotations
    └── instances_minival2014.json
images
    └── val2014
```

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)

设置可执行权限

```csharp
chmod +x benchmark.${arch}
```

##   2 离线推理
310上执行，执行时使npu-smi info查看设备状态，确保device空闲

 coco_imgs_path: images存放路径

 coco_anns_path: annotations存放路径

```csharp
bash test/pth2om.sh 
bash test/eval_acc_perf.sh --coco_imgs_path=/root/data/coco/images/ --coco_anns_path=/root/data/coco/annotations/
```
**评测结果：**

| 模型 | 官网pth精度                                                  | 310离线推理精度 | 基准性能 | 310性能 |
| ---- | ------------------------------------------------------------ | --------------- | -------- | ------- |
| M2Det bs1 | IoU=[0.50,0.95]:37.8% | IoU=[0.50,0.95]:37.8%  | 51.498fps   | 44.6152fps |
| M2Det bs16 | IoU=[0.50,0.95]:37.8% | IoU=[0.50,0.95]:37.8%   | 65.136fps   | 47.3184fps |

4.性能对比

```
bs1: 310/基准性能=44.6152/51.498=0.866倍  
bs16: 310/基准性能=47.3184/65.136=0.726倍
```
