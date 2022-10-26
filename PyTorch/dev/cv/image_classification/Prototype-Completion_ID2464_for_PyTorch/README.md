# Prototype Completion with Primitive Knowledge for Few-Shot Learning
This repository contains the code for the paper:
<br>
[**Prototype Completion with Primitive Knowledge for Few-Shot Learning**](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Prototype_Completion_With_Primitive_Knowledge_for_Few-Shot_Learning_CVPR_2021_paper.pdf)
<br>
Baoquan Zhang, Xutao Li, Yunming Ye, Zhichao Huang, Lisai Zhang
<br>
CVPR 2021
<p align='center'>
  <img src='algorithm.png' width="800px">
</p>

### Abstract

Few-shot learning is a challenging task, which aims to learn a classifier for novel classes with few examples. Pre-training based meta-learning methods effectively tackle the problem by pre-training a feature extractor and then fine-tuning it through the nearest centroid based meta-learning. However, results show that the fine-tuning step makes very marginal improvements. In this paper, 1) we figure out the key reason, i.e., in the pre-trained feature space, the base classes already form compact clusters while novel classes spread as groups with large variances, which implies that fine-tuning the feature extractor is less meaningful; 2) instead of fine-tuning the feature extractor, we focus on estimating more representative prototypes during meta-learning. Consequently, we propose a novel prototype completion based meta-learning framework. This framework first introduces primitive knowledge (i.e., class-level part or attribute annotations) and extracts representative attribute features as priors. Then, we design a prototype completion network to learn to complete prototypes with these priors. To avoid the prototype completion error caused by primitive knowledge noises or class differences, we further develop a Gaussian based prototype fusion strategy that combines the mean-based and completed prototypes by exploiting the unlabeled samples. Extensive experiments demonstrate that our method: (i) obtain more accurate prototypes; (ii) outperforms state-of-the-art techniques by $2\% \sim 9\%$ in terms of classification accuracy.

### Citation

If you use this code for your research, please cite our paper:
```
@inproceedings{zhang2021prototype,
	author    = {Zhang, Baoquan and Li, Xutao and Ye, Yunming and Huang, Zhichao and Zhang, Lisai},
	title     = {Prototype Completion With Primitive Knowledge for Few-Shot Learning},
	booktitle = {CVPR},
	year      = {2021},
	pages     = {3754-3762}
}
```

## Dependencies
* Python 3.6
* [PyTorch 1.1.0](http://pytorch.org)

## Usage

### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/zhangbq-research/Prototype_Completion_for_FSL.git
    cd Prototype_Completion_for_FSL
    ```
2. Download and decompress dataset files: [**miniImageNet**](https://mega.nz/#!rx0wGQyS!96sFlAr6yyv-9QQPCm5OBFbOm4XSD0t-HlmGaT5GaiE) (courtesy of [**Spyros Gidaris**](https://github.com/gidariss/FewShotWithoutForgetting))

3. For the dataset loader, specify the path to the directory. For example, in Prototype_Completion_for_FSL/data/mini_imagenet.py line 30:
    ```python
    _MINI_IMAGENET_DATASET_DIR = 'path/to/miniImageNet'
    ```

### Pre-training
1. To pre-train a feature extractor on miniImageNet and obtain a good representation for each image:
    ```bash
    python main.py --phase pretrain --gpu 0,1,2,3 --save-path "./experiments/meta_part_resnet12_mini" \
    --head CosineNet --network ResNet --pre_head LinearNet --dataset miniImageNet
    ```
   
2. You can experiment with varying classification head by changing '--pre_head' argument to LinearRotateNet.

### Construct primitive knowledge for all classes
Download the file of [**glove_840b_300d**](https://nlp.stanford.edu/data/glove.840B.300d.zip) and then perform
```bash
    python ./prior/make_miniimagenet_primitive_knowledge.py
```

### Extract prior information from primitive knowledge
```bash
    python main.py --phase savepart --gpu 0,1,2,3 --save-path "./experiments/meta_part_resnet12_mini" \
    --network ResNet --dataset miniImageNet
```

### Learn to complete prototype
1. To train ProtoComNet on 5-way 1-shot miniImageNet benchmark:
```bash
    python main.py --phase metainfer --gpu 0,1,2,3 --save-path "./experiments/meta_part_resnet12_mini" \
    --train-shot 1 --val-shot 1 --train-query 15 --val-query 15 --head FuseCosNet --network ResNet --dataset miniImageNet
```
2. To train ProtoComNet on 5-way 5-shot miniImageNet benchmark:
```bash
    python main.py --phase metainfer --gpu 0,1,2,3 --save-path "./experiments/meta_part_resnet12_mini" \
    --train-shot 5 --val-shot 5 --train-query 15 --val-query 15 --head FuseCosNet --network ResNet --dataset miniImageNet
```

### Meta-training
1. To jointly fine-tune feature extractor and ProtoComNet on 5-way 1-shot miniImageNet benchmark:
    ```bash
    python main.py --phase metatrain --gpu 0,1,2,3 --save-path "./experiments/meta_part_resnet12_mini" \
    --train-shot 1 --val-shot 1 --train-query 15 --val-query 15 --head FuseCosNet --network ResNet --dataset miniImageNet
    ```
2. To jointly fine-tune feature extractor and ProtoComNet on 5-way 5-shot miniImageNet benchmark:
    ```bash
    python main.py --phase metatrain --gpu 0,1,2,3 --save-path "./experiments/meta_part_resnet12_mini" \
    --train-shot 5 --val-shot 5 --train-query 15 --val-query 15 --head FuseCosNet --network ResNet --dataset miniImageNet
    ```    

### Meta-testing
1. To evaluate performance on 5-way 1-shot miniImageNet benchmark:
    ```bash
    python main.py --phase metatest --gpu 0,1,2,3 --save-path "./experiments/meta_part_resnet12_mini" \
    --train-shot 1 --val-shot 1 --train-query 15 --val-query 15 --head FuseCosNet --network ResNet --dataset miniImageNet
    ```
2. To evaluate performance on 5-way 5-shot miniImageNet benchmark:
    ```bash
    python main.py --phase metatest --gpu 0,1,2,3 --save-path "./experiments/meta_part_resnet12_mini" \
    --train-shot 5 --val-shot 5 --train-query 15 --val-query 15 --head FuseCosNet --network ResNet --dataset miniImageNet
    ```

## Acknowledgments

This code is based on the implementations of [**MetaOptNet**](https://github.com/kjunelee/MetaOptNet.git)
