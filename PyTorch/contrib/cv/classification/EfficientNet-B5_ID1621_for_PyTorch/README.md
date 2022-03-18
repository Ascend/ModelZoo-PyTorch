# EfficientNet-B5

This implements training of on the ImageNet dataset, mainly modified from https://github.com/facebookresearch/pycls

## EfficientNet-B5 Detail 

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 
Therefore, EfficientNet-B5 is re-implemented using semantics such as custom OP. For details, see  /pycls/pycls/models/effnet.py


## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- pip install pycls
- git clone https://github.com/facebookresearch/pycls
- pip install -r requirements.txt
- python setup.py develop --user
- mkdir -p /path/pycls/pycls/datasets/data
- ln -s /path/imagenet /path/pycls/pycls/datasets/data/imagenet
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training 

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash

# 1p train 1p
bash ./test/train_full_1p.sh  --data_path=data_path
bash ./test/train_performance_1p.sh --data_path=data_path

#  8p train 8p
bash ./test/train_full_8p.sh  --data_path=data_path
bash ./test/train_performance_8p.sh --data_path=data_path

# 8p eval 8p
bash ./test/train_eval_8p.sh --data_path=data_path

# online inference demo 
python3.7.5 demo.py

# To ONNX
python3.7.5 pthtar2onnx.py

```

## EfficientNet-B5 training result 

| Acc@1  | FPS  | Npu_nums | Epochs | AMP_Type |
| :----: | :--: | :------: | :----: | :------: |
|   -    | 47   |    1     |  100   |    O2    |
| 78.595 | 384  |    8     |  100   |    O2    |



