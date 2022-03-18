# Deep high-resolution representation learning for human pose estimation

## Introduction
```
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

#### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnet_w32](/configs/top_down/hrnet/coco/hrnet_w32_coco_256x192.py)  | 256x192 | 0.746 | 0.904 | 0.819 | 0.799 | 0.942 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192_20200708.log.json) |
| [pose_hrnet_w32](/configs/top_down/hrnet/coco/hrnet_w32_coco_384x288.py)  | 384x288 | 0.760 | 0.906 | 0.829 | 0.810 | 0.943 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_384x288-d9f0d786_20200708.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_384x288_20200708.log.json) |
| [pose_hrnet_w48](/configs/top_down/hrnet/coco/hrnet_w48_coco_256x192.py)  | 256x192 | 0.756 | 0.907 | 0.825 | 0.806 | 0.942 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192_20200708.log.json) |
| [pose_hrnet_w48](/configs/top_down/hrnet/coco/hrnet_w48_coco_384x288.py)  | 384x288 | 0.767 | 0.910 | 0.831 | 0.816 | 0.946 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288-314c8528_20200708.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_20200708.log.json) |

#### Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR  | Hand AP | Hand AR | Whole AP | Whole AR | ckpt | log |
| :---- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----: | :-----: | :------: |:-------: |:------: | :------: |
| [pose_hrnet_w32](/configs/top_down/hrnet/coco-wholebody/hrnet_w32_coco_wholebody_256x192.py)  | 256x192 | 0.700 | 0.746 | 0.567 | 0.645 | 0.637 | 0.688 | 0.473 | 0.546 | 0.553 | 0.626 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_256x192-853765cd_20200918.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_256x192_20200918.log.json) |
| [pose_hrnet_w32](/configs/top_down/hrnet/coco-wholebody/hrnet_w32_coco_wholebody_384x288.py)  | 384x288 | 0.701 | 0.773 | 0.586 | 0.692 | 0.727 | 0.783 | 0.516 | 0.604 | 0.586 | 0.674 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_384x288-78cacac3_20200922.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_384x288_20200922.log.json) |
| [pose_hrnet_w48](/configs/top_down/hrnet/coco-wholebody/hrnet_w48_coco_wholebody_256x192.py)  | 256x192 | 0.700 | 0.776 | 0.672 | 0.785 | 0.656 | 0.743 | 0.534 | 0.639 | 0.579 | 0.681 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_256x192-643e18cb_20200922.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_256x192_20200922.log.json) |
| [pose_hrnet_w48](/configs/top_down/hrnet/coco-wholebody/hrnet_w48_coco_wholebody_384x288.py)  | 384x288 | 0.722 | 0.790 | 0.694 | 0.799 | 0.777 | 0.834 | 0.587 | 0.679 | 0.631 | 0.716 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_20200922.log.json) |

#### Results on AIC val set.

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnet_w32](/configs/top_down/hrnet/aic/hrnet_w32_aic_256x192.py) | 256x192 | 0.675 | 0.957 | 0.751 | 0.703 | 0.961 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_aic_256x192-30a4e465_20200826.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_aic_256x192_20200826.log.json) |


#### Results on MPII val set.

| Arch  | Input Size | Mean | Mean@0.1   | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |
| [pose_hrnet_w32](/configs/top_down/hrnet/mpii/hrnet_w32_mpii_256x256.py) | 256x256 | 0.900 | 0.379 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256-6c4f923f_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256_20200812.log.json) |
| [pose_hrnet_w48](/configs/top_down/hrnet/mpii/hrnet_w48_mpii_256x256.py) | 256x256 | 0.900 | 0.383 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_mpii_256x256-92cab7bd_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_mpii_256x256_20200812.log.json) |


#### Results on CrowdPose test with [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) human detector.

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP (E) | AP (M) | AP (H) | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: | :------: |
| [pose_hrnet_w32](/configs/top_down/hrnet/crowdpose/hrnet_w32_crowdpose_256x192.py)  | 256x192 | 0.675 | 0.825 | 0.729 | 0.768 | 0.687 | 0.554 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_crowdpose_256x192-9b538d47_20201017.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_crowdpose_256x192_20201017.log.json) |


#### Results on PoseTrack2018 val with ground-truth bounding boxes.

| Arch  | Input Size | Head | Shou | Elb | Wri | Hip | Knee | Ankl | Total  | ckpt    | log     |
| :--- | :--------: | :------: |:------: |:------: |:------: |:------: |:------: | :------: | :------: |:------: |:------: |
| [pose_hrnet_w32](/configs/top_down/hrnet/posetrack18/hrnet_w32_posetrack18_256x192.py) | 256x192 | 87.4 | 88.6 | 84.3 | 78.5 | 79.7 | 81.8 | 78.8 | 83.0 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_256x192-1ee951c4_20201028.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_256x192_20201028.log.json) |

The models are first pre-trained on COCO dataset, and then fine-tuned on PoseTrack18.


#### Results on PoseTrack2018 val with [MMDetection](https://github.com/open-mmlab/mmdetection) pre-trained [Cascade R-CNN](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth) (X-101-64x4d-FPN) human detector.

| Arch  | Input Size | Head | Shou | Elb | Wri | Hip | Knee | Ankl | Total  | ckpt    | log     |
| :--- | :--------: | :------: |:------: |:------: |:------: |:------: |:------: | :------: | :------: |:------: |:------: |
| [pose_hrnet_w32](/configs/top_down/hrnet/posetrack18/hrnet_w32_posetrack18_256x192.py) | 256x192 | 78.0 | 82.9 | 79.5 | 73.8 | 76.9 | 76.6 | 70.2 | 76.9 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_256x192-1ee951c4_20201028.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_posetrack18_256x192_20201028.log.json) |

The models are first pre-trained on COCO dataset, and then fine-tuned on PoseTrack18.
