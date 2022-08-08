# Se_resnext50_32x4d

## ImageNet training with PyTorch

This implements training of Se_resnext50_32x4d on the ImageNet dataset, mainly modified from [Github](https://github.com/pytorch/examples/tree/master/imagenet)

## Se_resnext50_32x4d Detail
Can see in [Github](https://github.com/Cadene/pretrained-models.pytorch).

## Requirements
- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
  Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

## 1p training 1p
```
bash ./test/train_full_1p.sh  --data_path=数据集路径    # 精度训练
bash ./test/train_performance_1p.sh  --data_path=数据集路径  # 性能训练
```

## 8p training 8p
```
bash ./test/train_full_8p.sh  --data_path=数据集路径         # 精度训练
bash ./test/train_performance_8p.sh  --data_path=数据集路径  # 性能训练
```

## eval default 8p， should support 1p
`bash ./test/train_eval_8p.sh  --data_path=数据集路径`

## online inference demo
`python3.7.5 demo.py`

## To ONNX
`python3.7.5 pthtar2onx.py`
        

## Se_resnext50_32x4d training result

| Acc@1    | FPS       | Npu_nums| Epochs   | Type     |
| :------: | :------:  | :------ | :------: | :------: |
|     |  582      | 1       |   1    | O2       |
|  78.239  |  2953     | 8       | 100      | O2       |
