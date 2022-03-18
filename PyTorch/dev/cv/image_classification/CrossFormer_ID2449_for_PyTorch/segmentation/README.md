# CrossFormer Segmentation
Our semantic segmentation code is developed on top of [MMSegmentation v0.12.0]

For more details please refer to our paper [CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention]




## Prerequisites

1. Libraries (Python3.6-based)
```bash
pip3 install mmcv-full==1.2.7 mmsegmentation==0.12.0
```

2. Prepare ADE20K dataset according to guidelines in [MMSegmentation v0.12.0]

3. Prepare pretrained CrossFormer models
```python
import torch
ckpt = torch.load("crossformer-s.pth") ## load classification checkpoint
torch.save(ckpt["model"], "backbone-corssformer-s.pth") ## only model weights are needed
```



## Getting Started

1. Modify `data_root` in `configs/_base_/datasets/ade20k.py`  and `configs/_base_/datasets/ade20k_swin.py` to your path to the ADE20K dataset.

2. Training
```bash
## Use config in Results table listed below as <CONFIG_FILE>
./dist_train.sh <CONFIG_FILE> <GPUS> <PRETRAIN_MODEL>

## e.g. train fpn_crossformer_b model with 8 GPUs
./dist_train.sh configs/fpn_crossformer_b_ade20k_40k.py 8 path/to/backbone-corssformer-s.pth
```

3. Inference
```bash
./dist_test.sh <CONFIG_FILE> <GPUS> <DET_CHECKPOINT_FILE>

## e.g. evaluate semantic segmentation model by mIoU
./dist_test.sh configs/fpn_crossformer_b_ade20k_40k.py 8 path/to/ckpt
```
**Notes:** We use single-scale testing by default, you can enable multi-scale testing or flip testing manually by following the instructions in `configs/_base_/datasets/ade20k[_swin].py`.




## Results

### Semantic FPN

| Backbone      | Iterations | Params | FLOPs | IOU | config| Models|
| ------------- | :-----: | ------:| -----:| ------:| :-----| :---------------|

### UPerNet

| Backbone      | Iterations | Params | FLOPs | IOU    | MS IOU | config| Models|
| ------------- | :--------: | ------:| -----:| ------:| ------:| :-----| :---------------|
| ResNet-101    | 160K   | 86.0M | 1029.0G | 44.9  | - | - | - |
| Swin-T        | 160K   | 60.0M | 945.0G  | 44.5  | 45.8 | - | - |

**Notes:**
- *MS IOU* means *IOU* with multi-scale testing.
- Models are trained on ADE20K. Backbones are initialized with weights pre-trained on ImageNet-1K.
- For Semantic FPN, models are trained for 80K iterations with batch size 16. For UperNet, models are trained for 160K iterations.
- More detailed training settings can be found in corresponding configs.
- More results can be seen in our paper.




## FLOPs and Params Calculation
use `get_flops.py` to calculate FLOPs and #parameters of the specified model.
```bash
python get_flops.py <CONFIG_FILE> --shape <height> <width>

## e.g. get FLOPs and #params of fpn_crossformer_b with input image size [1024, 1024]
python get_flops.py configs/fpn_crossformer_b_ade20k_40k.py --shape 1024 1024
```

**Notes:** Default input image size is [1024, 1024]. For calculation with different input image size, you need to change `<height> <width>` in the above command and change `img_size` in `crossformer_factory.py` accordingly at the same time.




## Citing Us

```
@article{wang2021crossformer,
  title={CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention},
  author={Wang, Wenxiao and Yao, Lu and Chen, Long and Cai, Deng and He, Xiaofei and Liu, Wei},
  journal={arXiv preprint arXiv:2108.00154},
  year={2021}
}
```
