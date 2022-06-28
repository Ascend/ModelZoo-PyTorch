# YOLOR 模型使用说明

## Requirements
* NPU配套的run包安装
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)
* (可选)参考《Pytorch 网络模型移植&训练指南》6.4.2章节，配置cpu为性能模式，以达到模型最佳性能；不开启不影响功能。

安装其他依赖（先安装NPU版本的pytorch和apex，再安装其他依赖）：
```
pip install -r requirements.txt
```
注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision 建议Pillow版本是9.1.0 torchvision版本是0.6.0
## Dataset
1. 下载coco数据集，包含图片、annotations、labels图片、annotations:
	```
	cd yolor
	bash scripts/get_coco.sh
	```
    coco目录结构如下：
	```
   coco
   |-- LICENSE
   |-- README.txt
   |-- annotations
   |   `-- instances_val2017.json
   |-- images
   |   |-- test2017
   |   |-- train2017
   |   `-- val2017
   |-- labels
   |   |-- train2017
   |   |-- train2017.cache3
   |   |-- val2017
   |   `-- val2017.cache3
   |-- test-dev2017.txt
   |-- train2017.cache
   |-- train2017.txt
   |-- val2017.cache
   `-- val2017.txt
	```

### NPU 1P：在目录yolor下，运行 train_performance_1p.sh  data_path为coco数据集的路径
```
chmod +x ./test/train_performance_1p.sh
bash ./test/train_performance_1p.sh  --data_path=/data/coco   #性能训练
```
若需要指定训练使用的卡号, 可修改train_performance_1p.sh文件 "--npu 0"配置项,其中卡号为0-7

### NPU 8P：在目录yolor下，运行 train_performance_8p.sh  data_path为coco数据集的路径
```
chmod +x ./test/train_performance_8p.sh
bash ./test/train_performance_8p.sh  --data_path=/data/coco   #性能训练
```


### NPU 8P Full：在目录yolor下，运行 train_full_8p.sh  data_path为coco数据集的路径
```
chmod +x ./test/train_full_8p.sh
bash ./test/train_full_8p.sh  --data_path=/data/coco        #精度训练
```

## Evaluation
复制训练好的last.pt到pretrained文件夹下，运行evaluation_npu.sh (npu) / evaluation_gpu.sh (gpu)
```
chmod +x ./test/evaluation_xxx.sh
bash ./test/evaluation_xxx.sh
```

## 迁移学习
参考https://github.com/WongKinYiu/yolor/issues/103，更改./cfg/yolo_p6.cfg中**对应行**的classes和filters：

以coco为例，原80类别现在改为81：classes = 81, filters = anchor * (5 + classes) = 3 * (5 + 81) = 258，更改后的.cfg命名为yolor_p6_finetune.cfg，复制训练好的last.pt到pretrained文件夹下，运行train_finetune_1p.sh
```
chmod +x ./test/train_finetune_1p.sh
bash ./test/train_finetune_1p.sh
```

## 白名单
### Transpose whilte list

路径：/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/impl/dynamic/transpose.py 
#120行左右
```
[8,3,160,160,85], [8,3,80,80,85], [8,3,40,40,85], [8,3,20,20,85], [8,3,85,160,160], [8,3,85,80,80]
```
### Slice_d whilte list
路径：/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/impl/slice_d.py 
#7500行左右
```
["float16", [32,3,96,168,85], [32,3,96,168,2]],
["float16", [32,3,96,168,85], [32,3,96,168,4]],
["float16", [32,3,80,168,85], [32,3,80,168,2]],
["float16", [32,3,80,168,85], [32,3,80,168,4]],
["float16", [32,3,48,84,85], [32,3,48,84,2]],
["float16", [32,3,48,84,85], [32,3,48,84,4]],
["float16", [32,3,40,84,85], [32,3,40,84,2]],
["float16", [32,3,40,84,85], [32,3,40,84,4]],
["float16", [32,3,24,42,85], [32,3,24,42,2]],
["float16", [32,3,24,42,85], [32,3,24,42,4]],
["float16", [32,3,20,42,85], [32,3,20,42,2]],
["float16", [32,3,20,42,85], [32,3,20,42,4]],
["float32", [8, 3, 160, 160, 85], [8, 3, 160, 160, 1]],
["float32", [8, 3, 80, 80, 85], [8, 3, 80, 80, 1]],
```