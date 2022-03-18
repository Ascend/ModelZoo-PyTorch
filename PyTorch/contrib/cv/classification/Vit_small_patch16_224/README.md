# Vit_small_patch16_224

This implements training of vit_small_patch16_224 on the ImageNet-2012 dataset, mainly modified from [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models).

## Vit_small_patch16_224 Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.Therefore, vit_small_patch16_224 is re-implemented using semantics such as custom OP.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))  

- `pip install -r requirements.txt`

- Download the Imagenet-2012 dataset. Refer to the original repository https://github.com/rwightman/pytorch-image-models


## Training

To train a model, run `train.py` with the desired model architecture and the path to the Imagenet-2012 dataset:

```bash
# training 1p accuracy
bash test/train_full_1p.sh --data_path=real_data_path

# training 1p performance
bash test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash test/train_performance_8p.sh --data_path=real_data_path

# finetune
bash test/train_finetune_1p.sh --data_path=real_data_path --weight=real_weight_path

# Online inference demo
python demo.py --checkpoint real_checkpoint_path

# To ONNX
python pthtar2onnx.py 
```

## Vit_small_patch16_224 training result


|        | top1  | AMP_Type | Epochs |   FPS   |
| :----: | :---: | :------: | :----: | :-----: |
| 1p-GPU |   -   |    O2    |   1    | 586.67  |
| 1p-NPU |   -   |    O2    |   1    | 304.06  |
| 8p-GPU | 67.65 |    O2    |  100   | 4556.28 |
| 8p-NPU | 67.67 |    O2    |  100   | 2373.80 |

