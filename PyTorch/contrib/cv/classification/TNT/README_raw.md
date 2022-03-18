# Transformer in Transformer (TNT)
By Kai Han, An Xiao, Enhua Wu, Jianyuan Guo, Chunjing Xu, Yunhe Wang. [[arXiv link]](https://arxiv.org/abs/2103.00112)

![image](https://user-images.githubusercontent.com/9500784/122160150-ff1bca80-cea1-11eb-9329-be5031bad78e.png)

## Requirements
Pytorch 1.7.0,
timm 0.3.2,
apex

## Code
- Training example for 8 GPUs:
```
python -m torch.distributed.launch --nproc_per_node=8 train.py /path/to/imagenet/ --model tnt_s_patch16_224 --sched cosine --epochs 300 --opt adamw -j 8 --warmup-lr 1e-6 --mixup .8 --cutmix 1.0 --model-ema --model-ema-decay 0.99996 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 5 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 1e-3 --weight-decay .05 --drop 0 --drop-path .1 -b 128 --output /path/to/save/models/
```

- Pretrained models

|Model|Params (M)|FLOPs (B)|Top-1|Top-5|URL|
|-|-|-|-|-|-|
|TNT-S|23.8|5.2|81.5|95.7|[[BaiduDisk]](https://pan.baidu.com/s/1AwJDWEPl-hqLHfUvqmlqxQ), Password: 7ndi|
|TNT-B|65.6|14.1|82.9|96.3|[[BaiduDisk]](https://pan.baidu.com/s/1_TemN7kvWuYeZohisObQ1w), Password: 2gb7|

- Evaluate example:
```
python train.py /path/to/imagenet/ --model tnt_s_patch16_224 -b 256 --pretrain_path /path/to/pretrained/model/ --evaluate
```

## Citation
```
@misc{han2021transformer,
      title={Transformer in Transformer}, 
      author={Kai Han and An Xiao and Enhua Wu and Jianyuan Guo and Chunjing Xu and Yunhe Wang},
      year={2021},
      eprint={2103.00112},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Third-party implementations
1. Pytorch (timm) with ImageNet pretrained models: https://www.github.com/rwightman/pytorch-image-models/tree/master/timm/models/tnt.py
2. Pytorch (mmclassification) with ImageNet pretrained models: https://github.com/open-mmlab/mmclassification/blob/master/docs/model_zoo.md
3. JAX/FLAX: https://github.com/NZ99/transformer_in_transformer_flax
4. MindSpore Code: https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/TNT and pretrained weights on Oxford-IIIT Pets dataset: https://www.mindspore.cn/resources/hub/details?noah-cvlab/gpu/1.1/tnt_v1.0_oxford_pets
