# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## Introduction

[ALGORITHM]

```latex
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

## Results and models

### ADE20K

| Method | Backbone | Crop Size | pretrain | pretrain img size | Batch Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(ms+flip) | config | download |
| ------ | -------- | --------- | ---------- | ------- | -------- | --- | --- | -------------- | ----- | ------------: | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| UperNet | Swin-T | 512x512 | ImageNet-1K | 224x224 | 16          | 160000   | 5.02        | 21.06              | 44.41 | 45.79            | [config](   )  | [model](   ) &#124; [log](   )     |
| UperNet | Swin-S | 512x512  | ImageNet-1K | 224x224 | 16          | 160000   | 6.17        | 14.72              | 47.72 | 49.24             | [config](   )  | [model](   ) &#124; [log](   )     |
| UperNet | Swin-B | 512x512 | ImageNet-1K | 224x224 | 16           | 160000   | 7.61        | 12.65              | 47.99 | 49.57             | [config](   )  | [model](   ) &#124; [log](   )     |
| UperNet | Swin-B | 512x512  | ImageNet-22K | 224x224 | 16          | 160000   | -        | -              | 50.31 | 51.9             | [config](   )  | [model](   ) &#124; [log](   )     |
| UperNet | Swin-B | 512x512  | ImageNet-1K | 384x384 | 16          | 160000   | 8.52        | 12.10              | 48.35 | 49.65             | [config](   )  | [model](   ) &#124; [log](   )     |
| UperNet | Swin-B | 512x512  | ImageNet-22K | 384x384 | 16          | 160000   | -        | -              | 50.76 | 52.4             | [config](   )  | [model](   ) &#124; [log](   )     |
