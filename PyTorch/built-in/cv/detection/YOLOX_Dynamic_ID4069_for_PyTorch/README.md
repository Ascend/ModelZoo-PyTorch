### 参考链接

https://github.com/Megvii-BaseDetection/YOLOX/tree/0.3.0

当前只做了最基本的功能迁移，默认开启模糊编译。

模糊编译配置见`yolox/core/trainer.py`。

混合精度使用apex的O1，loss_scale为1024。

### 数据集

使用VOC2012，放在datasets目录下。

|
----datasets
    |----VOCdevkit
         |----VOC2012
              |----Annotations
              |----ImageSets
              |----JPEGImages
              |----labels
              |----SegmentationClass
              |----SegmentationObject

### 执行训练

bash run1p.sh