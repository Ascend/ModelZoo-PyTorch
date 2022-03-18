# Inception_v4 

This implements training of inception_v4 on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).

## Inception_v4 Detail

Details, see ./inceptionv4.py


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

To train a model, run `main.py`or `main-8p.py` with the desired model architecture and the path to the ImageNet dataset:



# 1p prefomance training 1p
bash test/train_performance_1p.sh

# 8p prefomance training 8p
bash test/train_performance_8p.sh

# 1p full training 1p
bash test/train_performance_1p.sh

# 8p full training 8p
bash test/train_performance_8p.sh

# online inference demo 
python3.7.5 demo.py

# To ONNX
python3.7.5 pthtar2onnx.py

## Inception_v4 training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| 81.64    | 2126      | 8        | 215      | O2       |
