

## LV-ViT 

All Tokens Matter: Token Labeling for Training Better Vision Transformers ,based Transformer model for image classification, detail in  ([arxiv](https://arxiv.org/abs/2104.10858))

## Requirements

torch>=1.4.0
torchvision>=0.5.0
pyyaml
scipy
timm==0.4.5
Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
data prepare: ImageNet with the following folder structure

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Label generation

To generate token label data for training:

```bash
python3 generate_label.py /path/to/imagenet/train /path/to/save/label_top5_train_nfnet --model dm_nfnet_f6 --pretrained --img-size 576 -b 32 --crop-pct 1.0
```

also provided genarated labeled date in  [BaiDu Yun](https://pan.baidu.com/s/1YBqiNN9dAzhEXtPl61bZJw) (password: y6j2)

## Model Train

Train the LV-ViT-S: 

```python
1:train on 1 NPU
bash /test/train_full_1p.sh '/Path_to_Imagenet' 'Path_to_Token-label-data'
Example: bash /test/train_full_1p.sh '/opt/npu/imagenet/' './label_top5_train_nfnet'

2:train on 8 NPU
bash /test/train_full_8p.sh '/Path_to_Imagenet' 'Path_to_Token-label-data'
Example: bash /test/train_full_8p.sh '/opt/npu/imagenet/' './label_top5_train_nfnet'
```

Get model performance

```python
1:test 1p performance
bash test/train_performance_1p.sh '/Path_to_Imagenet/' '/Path_to_Token-label-data/'
Example: bash test/train_performance_1p.sh  '/opt/npu/imagenet/' './label_top5_train_nfnet'
2:test 8p performance
bash test/train_performance_8p.sh '/Path_to_Imagenet/' '/Path_to_Token-label-data/'
Example: bash test/train_performance_8p.sh '/opt/npu/imagenet/' './label_top5_train_nfnet'
```

## Validation

Replace DATA_DIR with your imagenet validation set path and MODEL_DIR with the checkpoint path
```python
bash test/train_eval_8p.sh '/PATHTO/imagenet/val' '/PATHTO/LVVIT/eval_pth' 
Example:test/train_eval_8p.sh '/opt/npu/imagenet/val' '/trained/model.pth.tar'
```

## Fine-tuning

To Fine-tune the pre-trained LV-ViT-S
```python
bash /test/train_finetune_1p.sh '/Path_to_Imagenet/' '/Path_to_Token-label-data/' '/Pah_to_Trained_pth/'
Example: bash /test/train_full_1p.sh '/opt/npu/imagenet/' './label_top5_train_nfnet' './finetune/lvvit_s-26m-224-83.3.pth.tar'
```



## About Train FPS

```yaml
Example log:Train: 257 [ 150/625 ( 24%)]  Loss:  9.841134 (10.1421)  Time: 1.941s, 1054.88/s  (2.048s, 1000.09/s)  LR: 4.609e-04  Data: 0.029 (0.062)
As log  above get FPS：1054.88
```

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md