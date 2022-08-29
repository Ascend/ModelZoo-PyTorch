### Install

进入到项目下面，执行下面的命令

```bash
pip install -r requirements.txt
pip install -v -e .
pip install cython 
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision ,建议Pillow版本是9.1.0 torchvision版本是0.6.0

### Training

模型存在动态Shape，为提升性能，固定shape使用分档策略，同时模型默认开启多尺度，故训练前期持续有算子编译，现象为iter_time抖动。性能数据请关注性能稳定之后（400step之后很少有算子编译）。精度测试须300epoch，前期性能波动对整体影响较小。

shell脚本会将传入的`data_path`软连接到`./datasets`目录下，默认使用VOC2012数据集，使用其它数据集须自行修改配置文件并将数据转为COCO格式。

注：压测后发现模型对随机种子敏感，使用不同种子最终精度会有明显抖动，甚至会有低概率mAP有20%以上抖动（竞品上有类似现象）。当前针对默认配置（VOC2012/yolox-s）固定了随机种子，保证结果可复现，若更换了模型配置或数据集，请自行修改相关参数。随机种子设置在`yolox/exp/base_exp.py`中设置。


```bash
# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path
```


### Result
默认配置（yolox-s + VOC2012）

| 名称   | 精度 | 性能    |
| ------ | ---- | ------- |
| A40-1p | -    | 37.62 fps |
| 910A-1p | -    | 54.05 fps |
| A40-8p | 0.410 | 285.85 fps|
| 910A-8p | 0.407 | 320 fps  |

可选配置（yolox-x + COCO 众智交付）

| 名称   | 精度 | 性能    |
| ------ | ---- | ------- |
| V00-1p | -    | 20 fps   |
| 910A-1p | -    | 20.5 fps |
| V100-8p | 50.7 | 106 fps  |
| 910A-8p | 50.5 | 140 fps  |

### Reference

url=https://github.com/Megvii-BaseDetection/YOLOX

branch=main

commit_id=dd5700c24693e1852b55ce0cb170342c19943d8b