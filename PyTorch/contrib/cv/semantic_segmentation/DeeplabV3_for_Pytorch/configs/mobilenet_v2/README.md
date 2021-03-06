# MobileNetV2: Inverted Residuals and Linear Bottlenecks

## Introduction

<!-- [ALGORITHM] -->

```latex
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
  year={2018}
}
```

## Results and models

### Cityscapes

| Method     | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                                   | download                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------- | -------- | --------- | ------: | -------: | -------------- | ----: | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FCN        | M-V2-D8  | 512x1024  |   80000 |      3.4 | 14.2           | 61.54 | -             | [config](   )           | [model](   ) &#124; [log](   )                                         |
| PSPNet     | M-V2-D8  | 512x1024  |   80000 |      3.6 | 11.2           | 70.23 | -             | [config](   )        | [model](   ) &#124; [log](   )                             |
| DeepLabV3  | M-V2-D8  | 512x1024  |   80000 |      3.9 | 8.4            | 73.84 | -             | [config](   )     | [model](   ) &#124; [log](   )                 |
| DeepLabV3+ | M-V2-D8  | 512x1024  |   80000 |      5.1 | 8.4            | 75.20 | -             | [config](   ) | [model](   ) &#124; [log](   ) |

### ADE20k

| Method     | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                               | download                                                                                                                                                                                                                                                                                                                                                                             |
| ---------- | -------- | --------- | ------: | -------: | -------------- | ----: | ------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| FCN        | M-V2-D8  | 512x512   |  160000 |      6.5 | 64.4           | 19.71 | -             | [config](   )           | [model](   ) &#124; [log](   )                                         |
| PSPNet     | M-V2-D8  | 512x512   |  160000 |      6.5 | 57.7           | 29.68 | -             | [config](   )        | [model](   ) &#124; [log](   )                             |
| DeepLabV3  | M-V2-D8  | 512x512   |  160000 |      6.8 | 39.9           | 34.08 | -             | [config](   )     | [model](   ) &#124; [log](   )                 |
| DeepLabV3+ | M-V2-D8  | 512x512   |  160000 |      8.2 | 43.1           | 34.02 | -             | [config](   ) | [model](   ) &#124; [log](   ) |
