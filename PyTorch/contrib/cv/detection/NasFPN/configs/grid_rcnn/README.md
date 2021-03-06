# Grid R-CNN

## Introduction

[ALGORITHM]

```latex
@inproceedings{lu2019grid,
  title={Grid r-cnn},
  author={Lu, Xin and Li, Buyu and Yue, Yuxin and Li, Quanquan and Yan, Junjie},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}

@article{lu2019grid,
  title={Grid R-CNN Plus: Faster and Better},
  author={Lu, Xin and Li, Buyu and Yue, Yuxin and Li, Quanquan and Yan, Junjie},
  journal={arXiv preprint arXiv:1906.05688},
  year={2019}
}
```

## Results and Models

| Backbone    | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:-----------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50        | 2x      | 5.1      | 15.0           | 40.4   | -      | -        |
| R-101       | 2x      | 7.0      | 12.6           | 41.5   | -      | -        |
| X-101-32x4d | 2x      | 8.3      | 10.8           | 42.9   | -      | -        |
| X-101-64x4d | 2x      | 11.3     | 7.7            | 43.0   | -      | -        |

**Notes:**

- All models are trained with 8 GPUs instead of 32 GPUs in the original paper.
- The warming up lasts for 1 epoch and `2x` here indicates 25 epochs.
