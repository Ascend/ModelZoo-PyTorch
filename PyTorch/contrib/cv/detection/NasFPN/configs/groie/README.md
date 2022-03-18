# GRoIE

## A novel Region of Interest Extraction Layer for Instance Segmentation

By Leonardo Rossi, Akbar Karimi and Andrea Prati from
[IMPLab](http://implab.ce.unipr.it/).

We provide configs to reproduce the results in the paper for
"*A novel Region of Interest Extraction Layer for Instance Segmentation*"
on COCO object detection.

## Introduction

[ALGORITHM]

This paper is motivated by the need to overcome to the limitations of existing
RoI extractors which select only one (the best) layer from FPN.

Our intuition is that all the layers of FPN retain useful information.

Therefore, the proposed layer (called Generic RoI Extractor - **GRoIE**)
introduces non-local building blocks and attention mechanisms to boost the
performance.

## Results and models

The results on COCO 2017 minival (5k images) are shown in the below table.
You can find
[here](https://drive.google.com/drive/folders/19ssstbq_h0Z1cgxHmJYFO8s1arf3QJbT)
the trained models.

### Application of GRoIE to different architectures

| Backbone  | Method            | Lr schd | box AP | mask AP |  Config | Download|
| :-------: | :--------------: | :-----: | :----: | :-----: | :-------:| :--------:|
| R-50-FPN  | Faster Original  |   1x    |  37.4  |         |          |           |
| R-50-FPN  | + GRoIE          |   1x    |  38.3  |         |          |           |
| R-50-FPN  | Grid R-CNN       |   1x    |  39.1  |         |          |           |
| R-50-FPN  | + GRoIE          |   1x    |        |         |          |           |
| R-50-FPN  | Mask R-CNN       |   1x    |  38.2  |  34.7   |          |           |
| R-50-FPN  | + GRoIE          |   1x    |  39.0  |  36.0   |          |           |
| R-50-FPN  | GC-Net           |   1x    |  40.7  |  36.5   |          |           |
| R-50-FPN  | + GRoIE          |   1x    |  41.0  |  37.8   |          |           |
| R-101-FPN | GC-Net           |   1x    |  42.2  |  37.8   |          |           |
| R-101-FPN | + GRoIE          |   1x    |        |         |          |           |

## Citation

If you use this work or benchmark in your research, please cite this project.

```latex
@misc{rossi2020novel,
    title={A novel Region of Interest Extraction Layer for Instance Segmentation},
    author={Leonardo Rossi and Akbar Karimi and Andrea Prati},
    year={2020},
    eprint={2004.13665},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Contact

The implementation of GROI is currently maintained by
[Leonardo Rossi](https://github.com/hachreak/).
