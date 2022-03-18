# Associative Embedding (AE) + HRNet

## Introduction
```
@inproceedings{newell2017associative,
  title={Associative embedding: End-to-end learning for joint detection and grouping},
  author={Newell, Alejandro and Huang, Zhiao and Deng, Jia},
  booktitle={Advances in neural information processing systems},
  pages={2277--2287},
  year={2017}
}
@inproceedings{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5693--5703},
  year={2019}
}
```

## Results and models

### 2d Human Pose Estimation

#### Results on COCO val2017 without multi-scale test

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [HRNet-w32](/configs/bottom_up/hrnet/coco/hrnet_w32_coco_512x512.py)  | 512x512 | 0.654 | 0.863 | 0.720 | 0.710 | 0.892 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512_20200816.log.json) |
| [HRNet-w48](/configs/bottom_up/hrnet/coco/hrnet_w48_coco_512x512.py)  | 512x512 | 0.665 | 0.860 | 0.727 | 0.716 | 0.889 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_512x512-cf72fcdf_20200816.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_512x512_20200816.log.json) |

#### Results on COCO val2017 with multi-scale test. 3 default scales ([2, 1, 0.5]) are used.

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [HRNet-w32](/configs/bottom_up/hrnet/coco/hrnet_w32_coco_512x512.py)  | 512x512 | 0.698 | 0.877 | 0.760 | 0.748 | 0.907 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512_20200816.log.json) |
| [HRNet-w48](/configs/bottom_up/hrnet/coco/hrnet_w48_coco_512x512.py)  | 512x512 | 0.712 | 0.880 | 0.771 | 0.757 | 0.909 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_512x512-cf72fcdf_20200816.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_512x512_20200816.log.json) |
