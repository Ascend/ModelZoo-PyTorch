# Expectation-Maximization Attention Networks for Semantic Segmentation

## Introduction

<!-- [ALGORITHM] -->

```latex
@inproceedings{li2019expectation,
  title={Expectation-maximization attention networks for semantic segmentation},
  author={Li, Xia and Zhong, Zhisheng and Wu, Jianlong and Yang, Yibo and Lin, Zhouchen and Liu, Hong},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={9167--9176},
  year={2019}
}
```

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                      | download                                                                                                                                                                                                                                                                                                                                                     |
| ------ | -------- | --------- | ------: | -------: | -------------- | ----: | ------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| EMANet | R-50-D8  | 512x1024  |   80000 |      5.4 | 4.58           | 77.59 | 79.44         | [config](   )  | [model](   ) &#124; [log](   )     |
| EMANet | R-101-D8 | 512x1024  |   80000 |      6.2 | 2.87           | 79.10 | 81.21         | [config](   ) | [model](   ) &#124; [log](   ) |
| EMANet | R-50-D8  | 769x769   |   80000 |      8.9 | 1.97           | 79.33 | 80.49         | [config](   )   | [model](   ) &#124; [log](   )         |
| EMANet | R-101-D8 | 769x769   |   80000 |     10.1 | 1.22           | 79.62 | 81.00         | [config](   )  | [model](   ) &#124; [log](   )     |
