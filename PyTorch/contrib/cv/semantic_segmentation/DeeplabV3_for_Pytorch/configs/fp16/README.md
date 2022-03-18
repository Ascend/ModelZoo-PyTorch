# Mixed Precision Training

## Introduction

<!-- [OTHERS] -->

```latex
@article{micikevicius2017mixed,
  title={Mixed precision training},
  author={Micikevicius, Paulius and Narang, Sharan and Alben, Jonah and Diamos, Gregory and Elsen, Erich and Garcia, David and Ginsburg, Boris and Houston, Michael and Kuchaiev, Oleksii and Venkatesh, Ganesh and others},
  journal={arXiv preprint arXiv:1710.03740},
  year={2017}
}
```

## Results and models

### Cityscapes

| Method     | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                                | download                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FCN        | R-101-D8 | 512x1024  |   80000 | 5.37     | 8.64           | 76.80 |             - | [config](   )           | [model](   ) &#124; [log](   )                                         |
| PSPNet     | R-101-D8 | 512x1024  |   80000 | 5.34     | 8.77           | 79.46 |             - | [config](   )        | [model](   ) &#124; [log](   )                             |
| DeepLabV3  | R-101-D8 | 512x1024  |   80000 | 5.75     | 3.86           | 80.48 |             - | [config](   )     | [model](   ) &#124; [log](   )                 |
| DeepLabV3+ | R-101-D8 | 512x1024  |   80000 | 6.35     | 7.87           | 80.46 |             - | [config](   ) | [model](   ) &#124; [log](   ) |
