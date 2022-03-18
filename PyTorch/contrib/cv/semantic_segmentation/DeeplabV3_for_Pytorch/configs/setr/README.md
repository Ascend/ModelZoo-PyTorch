# Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{zheng2020rethinking,
  title={Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers},
  author={Zheng, Sixiao and Lu, Jiachen and Zhao, Hengshuang and Zhu, Xiatian and Luo, Zekun and Wang, Yabiao and Fu, Yanwei and Feng, Jianfeng and Xiang, Tao and Torr, Philip HS and others},
  journal={arXiv preprint arXiv:2012.15840},
  year={2020}
}
```

## Results and models

### ADE20K

| Method | Backbone | Crop Size | Batch Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) | config                                                                                                                          | download                                                                                                                                                                                                                                                                                                                                                     |
| ------ | -------- | --------- | ---------- | ------- | -------- | -------------- | ----- | ------------: | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| SETR-Naive | ViT-L | 512x512  | 16          | 160000   | 18.40        | 4.72              | 48.28 |             49.56 | [config](   )  | [model](   ) &#124; [log](   )     |
| SETR-PUP | ViT-L | 512x512  | 16          | 160000   | 19.54        | 4.50              | 48.24 |             49.99 | [config](   )  | [model](   ) &#124; [log](   )     |
| SETR-MLA | ViT-L | 512x512  | 8           | 160000   | 10.96        | -              | 47.34 |             49.05 | [config](   )  | [model](   ) &#124; [log](   )     |
| SETR-MLA | ViT-L | 512x512  | 16          | 160000   | 17.30        | 5.25              | 47.54 |             49.37 | [config](   )  | [model](   ) &#124; [log](   )     |
