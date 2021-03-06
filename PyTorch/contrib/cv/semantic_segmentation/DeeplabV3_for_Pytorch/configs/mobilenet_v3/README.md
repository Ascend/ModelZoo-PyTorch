# Searching for MobileNetV3

## Introduction

<!-- [ALGORITHM] -->

```latex
@inproceedings{Howard_2019_ICCV,
  title={Searching for MobileNetV3},
  author={Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and Le, Quoc V. and Adam, Hartwig},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  pages={1314-1324},
  month={October},
  year={2019},
  doi={10.1109/ICCV.2019.00140}}
}
```

## Results and models

### Cityscapes

| Method | Backbone           | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                                      | download                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------ | ------------------ | --------- | ------: | -------: | -------------- | ----: | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LRASPP | M-V3-D8            | 512x1024  |  320000 |      8.9 | 15.22          | 69.54 | 70.89         | [config](   )          | [model](   ) &#124; [log](   )                                     |
| LRASPP | M-V3-D8 (scratch)  | 512x1024  |  320000 |      8.9 | 14.77          | 67.87 | 69.78         | [config](   )  | [model](   ) &#124; [log](   )     |
| LRASPP | M-V3s-D8           | 512x1024  |  320000 |      5.3 | 23.64          | 64.11 | 66.42         | [config](   )         | [model](   ) &#124; [log](   )                                 |
| LRASPP | M-V3s-D8 (scratch) | 512x1024  |  320000 |      5.3 | 24.50          | 62.74 | 65.01         | [config](   ) | [model](   ) &#124; [log](   ) |
