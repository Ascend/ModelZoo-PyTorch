# CCNet: Criss-Cross Attention for Semantic Segmentation

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{huang2018ccnet,
    title={CCNet: Criss-Cross Attention for Semantic Segmentation},
    author={Huang, Zilong and Wang, Xinggang and Huang, Lichao and Huang, Chang and Wei, Yunchao and Liu, Wenyu},
    booktitle={ICCV},
    year={2019}
}
```

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                    | download                                                                                                                                                                                                                                                                                                                                               |
| ------ | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CCNet  | R-50-D8  | 512x1024  |   40000 | 6        | 3.32           | 77.76 |         78.87 | [config](   )  | [model](   ) &#124; [log](   )     |
| CCNet  | R-101-D8 | 512x1024  |   40000 | 9.5      | 2.31           | 76.35 |         78.19 | [config](   ) | [model](   ) &#124; [log](   ) |
| CCNet  | R-50-D8  | 769x769   |   40000 | 6.8      | 1.43           | 78.46 |         79.93 | [config](   )   | [model](   ) &#124; [log](   )         |
| CCNet  | R-101-D8 | 769x769   |   40000 | 10.7     | 1.01           | 76.94 |         78.62 | [config](   )  | [model](   ) &#124; [log](   )     |
| CCNet  | R-50-D8  | 512x1024  |   80000 | -        | -              | 79.03 |         80.16 | [config](   )  | [model](   ) &#124; [log](   )     |
| CCNet  | R-101-D8 | 512x1024  |   80000 | -        | -              | 78.87 |         79.90 | [config](   ) | [model](   ) &#124; [log](   ) |
| CCNet  | R-50-D8  | 769x769   |   80000 | -        | -              | 79.29 |         81.08 | [config](   )   | [model](   ) &#124; [log](   )         |
| CCNet  | R-101-D8 | 769x769   |   80000 | -        | -              | 79.45 |         80.66 | [config](   )  | [model](   ) &#124; [log](   )     |

### ADE20K

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                | download                                                                                                                                                                                                                                                                                                                               |
| ------ | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CCNet  | R-50-D8  | 512x512   |   80000 | 8.8      | 20.89          | 41.78 |         42.98 | [config](   )   | [model](   ) &#124; [log](   )         |
| CCNet  | R-101-D8 | 512x512   |   80000 | 12.2     | 14.11          | 43.97 |         45.13 | [config](   )  | [model](   ) &#124; [log](   )     |
| CCNet  | R-50-D8  | 512x512   |  160000 | -        | -              | 42.08 |         43.13 | [config](   )  | [model](   ) &#124; [log](   )     |
| CCNet  | R-101-D8 | 512x512   |  160000 | -        | -              | 43.71 |         45.04 | [config](   ) | [model](   ) &#124; [log](   ) |

### Pascal VOC 2012 + Aug

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                 | download                                                                                                                                                                                                                                                                                                                                   |
| ------ | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CCNet  | R-50-D8  | 512x512   |   20000 | 6        | 20.45          | 76.17 |         77.51 | [config](   )  | [model](   ) &#124; [log](   )     |
| CCNet  | R-101-D8 | 512x512   |   20000 | 9.5      | 13.64          | 77.27 |         79.02 | [config](   ) | [model](   ) &#124; [log](   ) |
| CCNet  | R-50-D8  | 512x512   |   40000 | -        | -              | 75.96 |         77.04 | [config](   )  | [model](   ) &#124; [log](   )     |
| CCNet  | R-101-D8 | 512x512   |   40000 | -        | -              | 77.87 |         78.90 | [config](   ) | [model](   ) &#124; [log](   ) |
