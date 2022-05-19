# Inception V4

This implements training of Inception V4 on the ImageNet dataset, mainly modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py

## Inception V4 Detail 

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 
Therefore,Inception V4 is re-implemented using semantics such as custom OP. For details, see https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionv4.py


## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- pip install -r requirements.txt
- git clone https://github.com/Cadene/pretrained-models.pytorch.git
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

# online inference demo 
python3.7.5 demo.py

# To ONNX
python3.7.5 pthtar2onnx.py

## Inception V4 training result 

|  Acc@1  |   FPS    | Npu_nums | Epochs | AMP_Type |
| :-----: | :------: | :------: | :----: | :------: |
|    -    |  226.677 |    1     |  2     |    O2    |
| 79.5130 | 1936.630 |    8     |  240   |    O2    |
