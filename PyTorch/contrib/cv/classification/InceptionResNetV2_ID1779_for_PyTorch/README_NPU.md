# Inception-ResNet-V2

This implements training of Inception-ResNet-V2 on the ImageNet dataset, mainly modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py

## Inception-ResNet-V2 Detail 

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 
Therefore,Inception-ResNet-V2  is re-implemented using semantics such as custom OP. For details, see https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionresnetv2.py


## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- pip install -r requirements.txt
  Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
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
python3 demo.py

# To ONNX
python3 pthtar2onnx.py

## Inception-ResNet-V2 training result 

|  Acc@1  |   FPS    | Npu_nums | Epochs | AMP_Type |
| :-----: | :------: | :------: | :----: | :------: |
|    -    | 228.975  |    1     |   2    |    O2    |
| 79.4250 | 1723.739 |    8     |  240   |    O2    |
