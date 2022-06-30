# RefineNet
Pytorch refinenet for segmentation and pytorch -> onnx -> tensorrt.

The goal of this project is to make a real-time light-weight semantic segmentation.

| Model   | FPS  |
| ------- | ---- |
| Pytorch | 5    |
| FP32    | 27   |
| FP16    | 33   |

This repository is heavily relyed on [light-weight-refinenet](https://github.com/DrSleep/light-weight-refinenet).

I make some changes for convenient to convert it to TensorRT engine.

## dataset

Models is trained on  [helen](http://www.ifp.illinois.edu/~vuongle2/helen/) dataset, the processed 11 class segmentation dataset is access to : https://pan.baidu.com/s/1blYDavW-TUIqjHH0ULdYnA code: 13dk

## Train

Build the helper code for calculating mean IoU written in Cython. For that, execute the following `python src/setup.py build_ext --build-lib=./src/`.

train with resnet50 backbone:

```
python src/train.py --enc 50
```

## Demo

see src/demo.py

![](./image/face_seg.jpg)

## Onnx

see src/onnx_export.py

## TensorRT

see https://github.com/midasklr/RefineNet_TensorRT

