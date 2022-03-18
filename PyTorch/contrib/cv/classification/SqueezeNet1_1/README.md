# Squeezenet1_1

This implements training of Squeezenet1_1 on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet). 

## Squeezenet1_1 Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 
Therefore, Squeezenet1_1 is re-implemented using semantics such as custom OP. For details, see models/Squeezenet.py . 


## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training 

To train a model, run `main.py or main_8p.py` with the desired model architecture and the path to the ImageNet dataset:

```bash

# O2 training 1p
bash scripts/run_1p.sh

# O2 training 8p
bash scripts/run_8p.sh
```

## Squeezenet1_1 training result 

| Acc@1       | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------:    | :------:  | :------: | :------: | :------: |
| -           | 384       | 1        | 240      | O2       |
| 58.54       | 1963      | 8        | 240      | O2       |



