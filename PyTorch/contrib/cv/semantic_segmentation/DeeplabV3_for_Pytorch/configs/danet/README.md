# Dual Attention Network for Scene Segmentation

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{fu2018dual,
  title={Dual Attention Network for Scene Segmentation},
  author={Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang,and Hanqing Lu},
  booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                    | download                                                                                                                                                                                                                                                                                                                                               |
| ------ | -------- | --------- | ------: | -------- | -------------- | ----: | ------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| DANet  | R-50-D8  | 512x1024  |   40000 | 7.4      | 2.66           | 78.74 | -             | [config](   )  | [model](   ) &#124; [log](   )     |
| DANet  | R-101-D8 | 512x1024  |   40000 | 10.9     | 1.99           | 80.52 | -             | [config](   ) | [model](   ) &#124; [log](   ) |
| DANet  | R-50-D8  | 769x769   |   40000 | 8.8      | 1.56           | 78.88 | 80.62         | [config](   )   | [model](   ) &#124; [log](   )         |
| DANet  | R-101-D8 | 769x769   |   40000 | 12.8     | 1.07           | 79.88 | 81.47         | [config](   )  | [model](   ) &#124; [log](   )     |
| DANet  | R-50-D8  | 512x1024  |   80000 | -        | -              | 79.34 | -             | [config](   )  | [model](   ) &#124; [log](   )     |
| DANet  | R-101-D8 | 512x1024  |   80000 | -        | -              | 80.41 | -             | [config](   ) | [model](   ) &#124; [log](   ) |
| DANet  | R-50-D8  | 769x769   |   80000 | -        | -              | 79.27 | 80.96         | [config](   )   | [model](   ) &#124; [log](   )         |
| DANet  | R-101-D8 | 769x769   |   80000 | -        | -              | 80.47 | 82.02         | [config](   )  | [model](   ) &#124; [log](   )     |

### ADE20K

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                | download                                                                                                                                                                                                                                                                                                                               |
| ------ | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DANet  | R-50-D8  | 512x512   |   80000 | 11.5     | 21.20          | 41.66 |         42.90 | [config](   )   | [model](   ) &#124; [log](   )         |
| DANet  | R-101-D8 | 512x512   |   80000 | 15       | 14.18          | 43.64 |         45.19 | [config](   )  | [model](   ) &#124; [log](   )     |
| DANet  | R-50-D8  | 512x512   |  160000 | -        | -              | 42.45 |         43.25 | [config](   )  | [model](   ) &#124; [log](   )     |
| DANet  | R-101-D8 | 512x512   |  160000 | -        | -              | 44.17 |         45.02 | [config](   ) | [model](   ) &#124; [log](   ) |

### Pascal VOC 2012 + Aug

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                 | download                                                                                                                                                                                                                                                                                                                                   |
| ------ | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| DANet  | R-50-D8  | 512x512   |   20000 | 6.5      | 20.94          | 74.45 |         75.69 | [config](   )  | [model](   ) &#124; [log](   )     |
| DANet  | R-101-D8 | 512x512   |   20000 | 9.9      | 13.76          | 76.02 |         77.23 | [config](   ) | [model](   ) &#124; [log](   ) |
| DANet  | R-50-D8  | 512x512   |   40000 | -        | -              | 76.37 |         77.29 | [config](   )  | [model](   ) &#124; [log](   )     |
| DANet  | R-101-D8 | 512x512   |   40000 | -        | -              | 76.51 |         77.32 | [config](   ) | [model](   ) &#124; [log](   ) |
