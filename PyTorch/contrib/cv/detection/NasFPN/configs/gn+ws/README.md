# Weight Standardization

## Introduction

[ALGORITHM]

```
@article{weightstandardization,
  author    = {Siyuan Qiao and Huiyu Wang and Chenxi Liu and Wei Shen and Alan Yuille},
  title     = {Weight Standardization},
  journal   = {arXiv preprint arXiv:1903.10520},
  year      = {2019},
}
```

## Results and Models

Faster R-CNN

| Backbone  | Style   | Normalization | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:---------:|:-------:|:-------------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| R-50-FPN  | pytorch | GN+WS         | 1x      | 5.9      | 11.7           | 39.7   | -       | -       | -       |
| R-101-FPN | pytorch | GN+WS         | 1x      | 8.9      | 9.0            | 41.7   | -       | -       | -       |
| X-50-32x4d-FPN | pytorch | GN+WS    | 1x      | 7.0      | 10.3           | 40.7   | -       | -       | -       |
| X-101-32x4d-FPN | pytorch | GN+WS   | 1x      | 10.8     | 7.6            | 42.1   | -       | -       | -       |

Mask R-CNN

| Backbone  | Style   | Normalization | Lr schd   | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:---------:|:-------:|:-------------:|:---------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| R-50-FPN  | pytorch | GN+WS         | 2x        | 7.3      | 10.5       | 40.6        | 36.6    | -     | -        |
| R-101-FPN | pytorch | GN+WS         | 2x        | 10.3     | 8.6        | 42.0        | 37.7    | -     | -        |
| X-50-32x4d-FPN | pytorch | GN+WS    | 2x        | 8.4      | 9.3        | 41.1        | 37.0    | -     | -        |
| X-101-32x4d-FPN | pytorch | GN+WS   | 2x        | 12.2     | 7.1        | 42.1        | 37.9    | -     | -        |
| R-50-FPN  | pytorch | GN+WS         | 20-23-24e | 7.3      | -          | 41.1        | 37.1    | -     | -        |
| R-101-FPN | pytorch | GN+WS         | 20-23-24e | 10.3     | -          | 43.1        | 38.6    | -     | -        |
| X-50-32x4d-FPN | pytorch | GN+WS    | 20-23-24e | 8.4      | -          | 42.1        | 38.0    | -     | -        |
| X-101-32x4d-FPN | pytorch | GN+WS   | 20-23-24e | 12.2     | -          | 42.7        | 38.5    | -     | -        |

Note:

- GN+WS requires about 5% more memory than GN, and it is only 5% slower than GN.
- In the paper, a 20-23-24e lr schedule is used instead of 2x.
- The X-50-GN and X-101-GN pretrained models are also shared by the authors.
