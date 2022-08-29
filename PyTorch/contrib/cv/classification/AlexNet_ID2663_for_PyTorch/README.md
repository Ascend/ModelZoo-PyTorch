# AlexNet

This implements training of AlexNet on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).

## AlexNet Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 
Therefore, AlexNet is re-implemented using semantics such as custom OP. For details, see main.py . 

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
  Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
  
## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

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


## AlexNet training result

| Acc@1    | Acc@5     | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------:  | :------: | :------: | :------: |
| -        | -         | 4581.735  | 1        | 1        | O2       |
|57.792%   | 80.418%   | 4046.933  | 8        | 90       | O2       |

> 注：
> 由于dropout算子存在一定的问题，这里的结果是：将dropout部分的代码使用cpu训练（这部分修改代码见alexnet.py文件），而其他部分代码使用npu训练所得到的结果。
精度达标，但由于代码中一部分使用cpu进行训练，性能降低至4046.933fps。

