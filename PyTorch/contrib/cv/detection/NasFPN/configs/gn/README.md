# Group Normalization

## Introduction

[ALGORITHM]

```latex
@inproceedings{wu2018group,
  title={Group Normalization},
  author={Wu, Yuxin and He, Kaiming},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2018}
}
```

## Results and Models

| Backbone      | model      | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:-------------:|:----------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| R-50-FPN (d)  | Mask R-CNN | 2x      | 7.1      | 11.0           | 40.2   | 36.4    | -      | -        |
| R-50-FPN (d)  | Mask R-CNN | 3x      | 7.1      | -              | 40.5   | 36.7    | -      | -        |
| R-101-FPN (d) | Mask R-CNN | 2x      | 9.9      | 9.0            | 41.9   | 37.6    | -      | -        |
| R-101-FPN (d) | Mask R-CNN | 3x      | 9.9      |                | 42.1   | 38.0    | -      | -        |
| R-50-FPN (c)  | Mask R-CNN | 2x      | 7.1      | 10.9           | 40.0   | 36.1    | -      | -        |
| R-50-FPN (c)  | Mask R-CNN | 3x      | 7.1      | -              | 40.1   | 36.2    | -      | -        |

**Notes:**

- (d) means pretrained model converted from Detectron, and (c) means the contributed model pretrained by [@thangvubk](https://github.com/thangvubk).
- The `3x` schedule is epoch [28, 34, 36].
- **Memory, Train/Inf time is outdated.**
