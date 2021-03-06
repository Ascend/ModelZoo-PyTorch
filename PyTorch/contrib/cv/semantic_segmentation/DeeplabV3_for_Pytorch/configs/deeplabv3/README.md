# Rethinking atrous convolution for semantic image segmentation

## Introduction

<!-- [ALGORITHM] -->

```latext
@article{chen2017rethinking,
  title={Rethinking atrous convolution for semantic image segmentation},
  author={Chen, Liang-Chieh and Papandreou, George and Schroff, Florian and Adam, Hartwig},
  journal={arXiv preprint arXiv:1706.05587},
  year={2017}
}
```

## Results and models

Note: `D-8` here corresponding to the output stride 8 setting for DeepLab series.

### Cityscapes

| Method    | Backbone        | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                                   | download                                                                                                                                                                                                                                                                                                                                                                                                   |
| --------- | --------------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DeepLabV3 | R-50-D8         | 512x1024  |   40000 | 6.1      | 2.57           | 79.09 |         80.45 | [config](   )         | [model](   ) &#124; [log](   )                                 |
| DeepLabV3 | R-101-D8        | 512x1024  |   40000 | 9.6      | 1.92           | 77.12 |         79.61 | [config](   )        | [model](   ) &#124; [log](   )                             |
| DeepLabV3 | R-50-D8         | 769x769   |   40000 | 6.9      | 1.11           | 78.58 |         79.89 | [config](   )          | [model](   ) &#124; [log](   )                                     |
| DeepLabV3 | R-101-D8        | 769x769   |   40000 | 10.9     | 0.83           | 79.27 |         80.11 | [config](   )         | [model](   ) &#124; [log](   )                                 |
| DeepLabV3 | R-18-D8         | 512x1024  |   80000 | 1.7      | 13.78          | 76.70 |         78.27 | [config](   )         | [model](   ) &#124; [log](   )                                 |
| DeepLabV3 | R-50-D8         | 512x1024  |   80000 | -        | -              | 79.32 |         80.57 | [config](   )         | [model](   ) &#124; [log](   )                                 |
| DeepLabV3 | R-101-D8        | 512x1024  |   80000 | -        | -              | 80.20 |         81.21 | [config](   )        | [model](   ) &#124; [log](   )                             |
| DeepLabV3 | R-18-D8         | 769x769   |   80000 | 1.9      | 5.55           | 76.60 |         78.26 | [config](   )          | [model](   ) &#124; [log](   )                                     |
| DeepLabV3 | R-50-D8         | 769x769   |   80000 | -        | -              | 79.89 |         81.06 | [config](   )          | [model](   ) &#124; [log](   )                                     |
| DeepLabV3 | R-101-D8        | 769x769   |   80000 | -        | -              | 79.67 |         80.81 | [config](   )         | [model](   ) &#124; [log](   )                                 |
| DeepLabV3 | R-101-D16-MG124 | 512x1024  |   40000 | 4.7      | - 6.96         | 76.71 |         78.63 | [config](   ) | [model](   ) &#124; [log](   ) |
| DeepLabV3 | R-101-D16-MG124 | 512x1024  |   80000 | -        | -              | 78.36 |         79.84 | [config](   ) | [model](   ) &#124; [log](   ) |
| DeepLabV3 | R-18b-D8        | 512x1024  |   80000 | 1.6      | 13.93          | 76.26 |         77.88 | [config](   )        | [model](   ) &#124; [log](   )                             |
| DeepLabV3 | R-50b-D8        | 512x1024  |   80000 | 6.0      | 2.74           | 79.63 |         80.98 | [config](   )        | [model](   ) &#124; [log](   )                             |
| DeepLabV3 | R-101b-D8       | 512x1024  |   80000 | 9.5      | 1.81           | 80.01 |         81.21 | [config](   )       | [model](   ) &#124; [log](   )                         |
| DeepLabV3 | R-18b-D8        | 769x769   |   80000 | 1.8      | 5.79           | 76.63 |         77.51 | [config](   )         | [model](   ) &#124; [log](   )                                 |
| DeepLabV3 | R-50b-D8        | 769x769   |   80000 | 6.8      | 1.16           | 78.80 |         80.27 | [config](   )         | [model](   ) &#124; [log](   )                                 |
| DeepLabV3 | R-101b-D8       | 769x769   |   80000 | 10.7     | 0.82           | 79.41 |         80.73 | [config](   )        | [model](   ) &#124; [log](   )                             |

### ADE20K

| Method    | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                        | download                                                                                                                                                                                                                                                                                                                                                       |
| --------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DeepLabV3 | R-50-D8  | 512x512   |   80000 | 8.9      | 14.76          | 42.42 |         43.28 | [config](   )   | [model](   ) &#124; [log](   )         |
| DeepLabV3 | R-101-D8 | 512x512   |   80000 | 12.4     | 10.14          | 44.08 |         45.19 | [config](   )  | [model](   ) &#124; [log](   )     |
| DeepLabV3 | R-50-D8  | 512x512   |  160000 | -        | -              | 42.66 |         44.09 | [config](   )  | [model](   ) &#124; [log](   )     |
| DeepLabV3 | R-101-D8 | 512x512   |  160000 | -        | -              | 45.00 |         46.66 | [config](   ) | [model](   ) &#124; [log](   ) |

### Pascal VOC 2012 + Aug

| Method    | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                         | download                                                                                                                                                                                                                                                                                                                                                           |
| --------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| DeepLabV3 | R-50-D8  | 512x512   |   20000 | 6.1      | 13.88          | 76.17 |         77.42 | [config](   )  | [model](   ) &#124; [log](   )     |
| DeepLabV3 | R-101-D8 | 512x512   |   20000 | 9.6      | 9.81           | 78.70 |         79.95 | [config](   ) | [model](   ) &#124; [log](   ) |
| DeepLabV3 | R-50-D8  | 512x512   |   40000 | -        | -              | 77.68 |         78.78 | [config](   )  | [model](   ) &#124; [log](   )     |
| DeepLabV3 | R-101-D8 | 512x512   |   40000 | -        | -              | 77.92 |         79.18 | [config](   ) | [model](   ) &#124; [log](   ) |

### Pascal Context

| Method    | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                               | download                                                                                                                                                                                                                                                                                                                                                                                   |
| --------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| DeepLabV3 | R-101-D8 | 480x480   |   40000 | 9.2      | 7.09           | 46.55 |         47.81 | [config](   ) | [model](   ) &#124; [log](   ) |
| DeepLabV3 | R-101-D8 | 480x480   |   80000 | -        | -              | 46.42 |         47.53 | [config](   ) | [model](   ) &#124; [log](   ) |

### Pascal Context 59

| Method    | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                               | download                                                                                                                                                                                                                                                                                                                                                                                   |
| --------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| DeepLabV3 | R-101-D8 | 480x480   |   40000 | -      | -           | 52.61 |         54.28 | [config](   ) | [model](   ) &#124; [log](   ) |
| DeepLabV3 | R-101-D8 | 480x480   |   80000 | -        | -              | 52.46 |         54.09 | [config](   ) | [model](   ) &#124; [log](   ) |
