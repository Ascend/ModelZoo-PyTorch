# High-resolution networks (HRNets) for object detection

## Introduction

[ALGORITHM]

```latex
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{SunZJCXLMWLW19,
  title={High-Resolution Representations for Labeling Pixels and Regions},
  author={Ke Sun and Yang Zhao and Borui Jiang and Tianheng Cheng and Bin Xiao
  and Dong Liu and Yadong Mu and Xinggang Wang and Wenyu Liu and Jingdong Wang},
  journal   = {CoRR},
  volume    = {abs/1904.04514},
  year={2019}
}
```

## Results and Models

### Faster R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :------:| :--------:|
|   HRNetV2p-W18  | pytorch |   1x    | 6.6      | 13.4           | 36.9   |         |  |
|   HRNetV2p-W18  | pytorch |   2x    | 6.6      |                | 38.9   |||
|   HRNetV2p-W32  | pytorch |   1x    | 9.0      | 12.4           | 40.2   | | |
|   HRNetV2p-W32  | pytorch |   2x    | 9.0        |              | 41.4   |||
|   HRNetV2p-W40  | pytorch |   1x    | 10.4     | 10.5           | 41.2   |  | |
|   HRNetV2p-W40  | pytorch |   2x    | 10.4     |                |  42.1  | |   |

### Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :------:|:------:|:--------:|
|   HRNetV2p-W18  | pytorch |   1x    | 7.0      | 11.7           | 37.7   | 34.2    | ||
|   HRNetV2p-W18  | pytorch |   2x    | 7.0      | -              | 39.8   | 36.0    |  | |
|   HRNetV2p-W32  | pytorch |   1x    | 9.4      | 11.3           | 41.2   | 37.1    |  | |
|   HRNetV2p-W32  | pytorch |   2x    | 9.4      | -              | 42.5   | 37.8    |  | |
|   HRNetV2p-W40  | pytorch |   1x    |  10.9    |                | 42.1   |  37.5   |   |  |
|   HRNetV2p-W40  | pytorch |   2x    |   10.9   |                | 42.8   |  38.2   |  |   |

### Cascade R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :------: | :--------: |
|   HRNetV2p-W18  | pytorch |   20e   |  7.0     | 11.0           | 41.2   ||  |
|   HRNetV2p-W32  | pytorch |   20e   |  9.4     | 11.0           | 43.3   |  | |
|   HRNetV2p-W40  | pytorch |   20e   |  10.8    |                | 43.8   |  | |

### Cascade Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :------:|:------:|:--------:|
|   HRNetV2p-W18  | pytorch |   20e   | 8.5      | 8.5            |41.6    |36.4     | | |
|   HRNetV2p-W32  | pytorch |   20e   |          | 8.3            |44.3    |38.6     |  |  |
|   HRNetV2p-W40  | pytorch |   20e   | 12.5     |                |45.1    |39.3     |   |     |

### Hybrid Task Cascade (HTC)

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :------:|:------:|:--------:|
|   HRNetV2p-W18  | pytorch |   20e   | 10.8     | 4.7            | 42.8   | 37.9    |  | |
|   HRNetV2p-W32  | pytorch |   20e   | 13.1     | 4.9            | 45.4   | 39.9    ||  |
|   HRNetV2p-W40  | pytorch |   20e   | 14.6     |                | 46.4   | 40.8    | |  |

### FCOS

| Backbone  | Style   |  GN     | MS train | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:-------:|:------:|:------:|:------:|:------:|:--------:|
|HRNetV2p-W18| pytorch | Y       | N       | 1x       | 13.0 | 12.9 | 35.3   | | |
|HRNetV2p-W18| pytorch | Y       | N       | 2x       | 13.0 | -    | 38.2   |  | |
|HRNetV2p-W32| pytorch | Y       | N       | 1x       | 17.5 | 12.9 | 39.5   ||  |
|HRNetV2p-W32| pytorch | Y       | N       | 2x       | 17.5 | -    | 40.8   |  | |
|HRNetV2p-W18| pytorch | Y       | Y       | 2x       | 13.0 | 12.9 | 38.3   |  ||  |
|HRNetV2p-W48| pytorch | Y       | Y       | 2x       | 20.3 | 10.8 | 42.7   | | |

**Note:**

- The `28e` schedule in HTC indicates decreasing the lr at 24 and 27 epochs, with a total of 28 epochs.
- HRNetV2 ImageNet pretrained models are in [HRNets for Image Classification](https://github.com/HRNet/HRNet-Image-Classification).
