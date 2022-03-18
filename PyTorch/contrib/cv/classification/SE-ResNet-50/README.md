# SE-ResNet-50

This implements training of SE-ResNet-50 on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).

## SE-ResNet-50 Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.Therefore, SE-ResNet-50 is re-implemented using semantics such as custom OP.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

# O2 training 1p
bash scripts/run_1p.sh

# O2 training 8p  
bash scripts/run_8p.sh

# O2 training 8p_eval
bash scripts/eval.sh

# O2 online inference demo
source scripts/set_npu_env.sh
python3.7.5 demo.py

# O2 To ONNX
source scripts/set_npu_env.sh
python3.7.5 pthtar2onnx.py


```

## SE-ResNet-50 training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 319.265   | 1        | 1        | O2       |
|          | 4126.888  | 8        | 100      | O2       |
