# Region Proposal by Guided Anchoring

## Introduction

[ALGORITHM]

We provide config files to reproduce the results in the CVPR 2019 paper for [Region Proposal by Guided Anchoring](https://arxiv.org/abs/1901.03278).

```latex
@inproceedings{wang2019region,
    title={Region Proposal by Guided Anchoring},
    author={Jiaqi Wang and Kai Chen and Shuo Yang and Chen Change Loy and Dahua Lin},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
    year={2019}
}
```

## Results and Models

The results on COCO 2017 val is shown in the below table. (results on test-dev are usually slightly higher than val).

| Method |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | AR 1000 | Config | Download |
| :----: | :-------------: | :-----: | :-----: | :------: | :------------: | :-----: | :------: | :--------: |
| GA-RPN |    R-50-FPN     |  caffe  |   1x    |   5.3    |      15.8      |  68.4   |   |   |
| GA-RPN |    R-101-FPN    |  caffe  |   1x    |   7.3    |      13.0      |  69.5   |  | |
| GA-RPN | X-101-32x4d-FPN | pytorch |   1x    |   8.5    |      10.0      |  70.6   |  |  |
| GA-RPN | X-101-64x4d-FPN | pytorch |   1x    |   7.1    |      7.5       |  71.2   | | |

|     Method     |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :------------: | :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
| GA-Faster RCNN |    R-50-FPN     |  caffe  |   1x    |   5.5    |                |  39.6  |          |            |
| GA-Faster RCNN |    R-101-FPN    |  caffe  |   1x    |   7.5    |                |  41.5  |          |            |
| GA-Faster RCNN | X-101-32x4d-FPN | pytorch |   1x    |   8.7    |      9.7       |  43.0  |          |            |
| GA-Faster RCNN | X-101-64x4d-FPN | pytorch |   1x    |   11.8   |      7.3       |  43.9  |          |            |
|  GA-RetinaNet  |    R-50-FPN     |  caffe  |   1x    |   3.5    |      16.8      |  36.9  |          |            |
|  GA-RetinaNet  |    R-101-FPN    |  caffe  |   1x    |   5.5    |      12.9      |  39.0  |          |            |
|  GA-RetinaNet  | X-101-32x4d-FPN | pytorch |   1x    |   6.9    |      10.6      |  40.5  |          |            |
|  GA-RetinaNet  | X-101-64x4d-FPN | pytorch |   1x    |   9.9    |      7.7       |  41.3  |          |            |

- In the Guided Anchoring paper, `score_thr` is set to 0.001 in Fast/Faster RCNN and 0.05 in RetinaNet for both baselines and Guided Anchoring.

- Performance on COCO test-dev benchmark are shown as follows.

|     Method     | Backbone  | Style | Lr schd | Aug Train | Score thr |  AP   | AP_50 | AP_75 | AP_small | AP_medium | AP_large | Download |
| :------------: | :-------: | :---: | :-----: | :-------: | :-------: | :---: | :---: | :---: | :------: | :-------: | :------: | :------: |
| GA-Faster RCNN | R-101-FPN | caffe |   1x    |     F     |   0.05    |       |       |       |          |           |          |          |
| GA-Faster RCNN | R-101-FPN | caffe |   1x    |     F     |   0.001   |       |       |       |          |           |          |          |
|  GA-RetinaNet  | R-101-FPN | caffe |   1x    |     F     |   0.05    |       |       |       |          |           |          |          |
|  GA-RetinaNet  | R-101-FPN | caffe |   2x    |     T     |   0.05    |       |       |       |          |           |          |          |
