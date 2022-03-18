# Mnasnet1_0

## ImageNet training with PyTorch

This implements training of ShuffleNetV1 on the ImageNet dataset, mainly modified from [Github](https://github.com/pytorch/examples/tree/master/imagenet).

## Mnasnet Detail

Base version of the model from [pytorch.torchvision](https://github.com/pytorch/vision/blob/master/torchvision/models/mnasnet.py).
The training script is adapted from [training script on imagenet](https://github.com/pytorch/examples/blob/master/imagenet/main.py).

## Requirements

- pytorch_ascend, apex_ascend, tochvision
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).

## Training

To train a model, run `train.py` with the desired model architecture and the path to the ImageNet dataset:

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
python3.7.5 demo.py

# To ONNX
python3.7.5 pthtar2onnx.py

## Mnasnet1_0 training result

| Acc@1    | FPS       | Npu_nums| Epochs   | Type     |
| :------: | :------:  | :------ | :------: | :------: |
| -        | 173       | 1       | 1        | O2       |
| 73.045   | 9188      | 8       | 300      | O1       |
