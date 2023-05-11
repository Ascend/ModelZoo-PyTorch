# Xception

This implements training of xception on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).

## Xception Detail
details, see ./xception.py


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script]

## Training

To train a model, run `main.py` or `main-8p.py` with the desired model architecture and the path to the ImageNet dataset:

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

# online inference demo 
python3 demo.py

# To ONNX
python3 pthtar2onnx.py

## Xception training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 492       | 1        | 150      | O2       |
| 78.86   | 1420      | 8        | 150      | O2       |
