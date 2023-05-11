# SE-ResNet-50

This implements training of SE-ResNet-50 on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).

## SE-ResNet-50 Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.Therefore, SE-ResNet-50 is re-implemented using semantics such as custom OP.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip3.7 install -r requirements.txt`
    Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision
    Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

# O2 training 1p
bash test/train_full_1p.sh --data_path=数据集路径

# O2 training 8p  
bash test/train_full_8p.sh --data_path=数据集路径

# O2 training 8p_eval
bash test/train_eval_8p.sh --data_path=数据集路径
- If you want to use custom weight for infering, modify the parameter '--resume=/THE/PATH/OF/CUSTOM/WEIGHT/'

# O2 online inference demo
source test/env_npu.sh
python3 demo.py

# O2 To ONNX
source test/env_npu.sh
python3 pthtar2onnx.py


```

## SE-ResNet-50 training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 319.265   | 1        | 1        | O2       |
|          | 4126.888  | 8        | 100      | O2       |