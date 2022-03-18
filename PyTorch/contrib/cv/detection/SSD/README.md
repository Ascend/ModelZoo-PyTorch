# SSD
本项目实现了 SSD (Single Shot MultiBox Detector) 在 NPU 上的训练，迁移自 [MMDetection](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd)，[MMCV](https://github.com/open-mmlab/mmcv/)。

## SSD Detail
本项目对于 MMDetection 和 MMCV 做了如下更改：
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

### 安装 MMCV
```
git clone -b v1.2.7 git://github.com/open-mmlab/mmcv.git

cp -f mmcv_need/_functions.py mmcv/mmcv/parallel/
cp -f mmcv_need/builder.py mmcv/mmcv/runner/optimizer/
cp -f mmcv_need/data_parallel.py mmcv/mmcv/parallel/
cp -f mmcv_need/dist_utils.py mmcv/mmcv/runner/
cp -f mmcv_need/distributed.py mmcv/mmcv/parallel/
cp -f mmcv_need/optimizer.py mmcv/mmcv/runner/hooks/

cd mmcv
export MMCV_WITH_OPS=1 
export MAX_JOBS=8
python3.7 setup.py build_ext
python3.7 setup.py develop
pip3.7 list | grep mmcv
cd ..
```

### 安装 MMDetection
```
pip3.7 install -r requirements/build.txt
pip3.7 install -v -e .
pip3.7 list | grep mmdet
```

### 准备数据集
1. 下载 COCO 数据集；
2. 创建 data 一级目录；
3. 将 COCO 数据集放于 data 目录下，形式如下。
```
SSD
├── configs
├── data
│   └── coco
│       ├── annotations
│       ├── train2017
│       ├── val2017
│       └── test2017
├── mmcv
├── mmdet
├── tools
```

## Training
### 单卡训练
```
bash test/train_full_1p.sh --data_path=数据集路径
```

### 8 卡训练
```
bash test/train_full_1p.sh --data_path=数据集路径
```

```

### 8 卡评估
```
bash test/train_full_1p.sh --data_path=数据集路径
```

### Demo
```
python3.7 demo.py
```

### 转换到 ONNX
```
python3.7 pthtar2onnx.py
```


```
### 日志路径
```
训练日志路径：网络脚本test下output文件夹内。例如：

      test/output/devie_id/train_${device_id}.log           # 训练脚本原生日志
      test/output/devie_id/SSD_bs8_1p_perf.log  # 1p性能训练结果日志
```

## SSD Training Result
| 0.5:0.95 mAP | FPS            | Npu_nums | Epochs   | AMP_Type |
| :----------: | :------------: | :------: | :------: | :------: |
| -            | 20.6 (X86 CPU) | 1        | 1        | O1       |
| 25.5         | 64.5 (ARM CPU) | 8        | 24       | O1       |

## Else
FPS 可使用 calc_fps.py 计算，使用方法为：
```
python3.7 calc_fps.py xxx.log.json ${gpu_nums} ${batch_size}
```
SSD 的 batch_size 为 8。

