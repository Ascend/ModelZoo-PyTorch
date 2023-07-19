# CSP_resnext50-mish

This implements training of csp_resnext50-mish on the ImageNet dataset, mainly modified from https://github.com/rwightman/pytorch-image-models.

## CSP_resnext50-mish Detail

For details, see ./timm/models/cspnet.py


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

To train a model, run `train_1p.py` or `train_8p.py` with the desired model architecture and the path to the ImageNet dataset:

# 1p prefomance training 1p
bash test/train_performance_1p.sh  --data_path=xxx

# 8p prefomance training 8p
bash test/train_performance_8p.sh  --data_path=xxx

# 1p full training 1p
bash test/train_performance_1p.sh  --data_path=xxx

# 8p full training 8p
bash test/train_performance_8p.sh  --data_path=xxx

# eval default 8p
bash ./test/train_eval_8p.sh  --data_path=xxx

#To ONNX
python3 pthtar2onx.py  --model-path path/to/model_best.pth.tar 

# online inference demo 
python3 demo.py --model-path /path/to/model_best.pth.tar
```

## CSP_resnext50-mish training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 202       | 1        | 1      | O2       |
| 79.36   | 1807      | 8        | 150      | O2       |
```

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md