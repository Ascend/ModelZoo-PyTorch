# EfficientNet-B3

This implements training of Efficientnet-B3 on the ImageNet dataset, mainly modified from [pycls](https://github.com/facebookresearch/pycls).

## EfficientNet-B3 Detail 

For details, see[pycls](https://github.com/facebookresearch/pycls).


## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- pip install pycls
- git clone https://github.com/facebookresearch/pycls
- pip install -r requirements.txt
- modify path of dataset in pycls/datasets/loader.py, you can modify the variable _DATA_DIR to your path of imagenet dataset.

## Training 

To train a model, run scripts with the desired model architecture and the path to the ImageNet dataset:

```bash
# 1p train 1p
bash test/train_full_1p.sh --data_path={data/path} # train accuracy

bash test/train_performance_1p.sh --data_path={data/path} # train performance

#  8p train 8p
bash test/train_full_8p.sh --data_path={data/path} # train accuracy

bash test/train_performance_8p.sh --data_path={data/path} # train performance

# 1p eval 1p
bash test/train_eval_8p.sh --data_path={data/path}

# online inference demo 
python3 demo.py

# To ONNX
python3 pthtar2onnx.py

```

## EfficientNet-B3 training result 

| Acc@1  | FPS  | Npu_nums | Epochs | AMP_Type |
| :----: | :--: | :------: | :----: | :------: |
|   -    | 267   |    1     |  100   |    O2    |
| 77.3418 | 1558  |    8     |  100   |    O2    |



