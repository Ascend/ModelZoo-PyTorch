# M2Det

note
- please download utils/pycocotools from origin repo:
- https://github.com/VDIGPKU/M2Det/tree/master/utils/pycocotools


本项目实现了 M2Det 在 NPU 上的训练，迁移自 [M2Det](https://github.com/qijiezhao/M2Det)。

## M2Det Detail
本项目对于 M2Det 做了如下更改：
1. 将设备从 CUDA 迁移到 NPU 上；
2. 使用 Apex 对原始代码进行修改，使用混合精度训练；
3. 对于一些操作，使用 NPU 算子优化性能，同时将一些操作转移到 CPU 上进行。

## Requirements
- NPU 配套的 run 包安装
- Python 3.7.5
- PyTorch (NPU 版本)
- Apex (NPU 版本)

### 导入环境变量
```
source test/env_npu.sh
```

### 安装 M2Det
```
pip3.7 install -r requirements.txt

```
### 获取权重文件

权重文件放在M2Det/weights目录下

[m2det512_vgg.pth](https://pan.baidu.com/s/1LDkpsQfpaGq_LECQItxRFQ)

[vgg16_reducedfc.pth](https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth)

### 准备数据集

1. 获取train2014数据集，获取方法参考github中[readme](https://github.com/qijiezhao/M2Det) 的方法获取
   
2. 创建目录

```
/root/data
      ├── coco_cache
      └── coco
          ├── annotations
              ├── instances_train2014.json
              ├── instances_val2014.json
              ├── instances_minival2014.json
              └── instances_valminusminival2014.json
          └── images
              ├── train2014
              └── val2014
```
### 编译初始化
进入模型所在目录M2Det下，然后执行
```
cd utils
bash make.sh
```

## Training
### 单卡训练
```
bash test/train_full_1p.sh
```

### 8 卡训练
```
bash test/train_full_8p.sh
```

### 8 卡评估
评估前需要先跑完bash test/train_full_8p.sh指令
```
bash test/train_eval_8p.sh
```

### 单卡性能
```
bash test/train_performance_1p.sh
```

###  8 卡性能
```
bash test/train_performance_8p.sh
```

### Demo
```
bash test/demo.sh
```

### finetuning
```
bash test/train_finetune_1p.sh --device_id=0 --pth_path=weights/M2Det_COCO_size512_netvgg16_epoch150.pth
```


```

## M2Det Training Result
| 0.5:0.95 mAP |   FPS   | Npu_nums | Epochs   | AMP_Type |  Loss Scale  |
| :----------: | :-----: | :------: | :------: | :------: | :----------: |
| -            |  4.638  | 1        | 1        | O1       | 32.0         |
| 30.4         |  60.997 | 8        | 160      | O1       | 32.0         |

M2Det 的 batch_size 为 16。

```

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md