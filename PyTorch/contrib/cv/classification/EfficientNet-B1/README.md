# EfficientNet-B1

This implements training of Efficientnet-B1 on the ImageNet dataset, mainly modified from [pycls](https://github.com/facebookresearch/pycls).

## EfficientNet-B1 Detail 

For details, see[pycls](https://github.com/facebookresearch/pycls).


## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- apt install bc
- python3 -m pip install --upgrade Pillow
- git clone https://github.com/facebookresearch/pycls
- pip install -r requirements.txt
  `Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0`
- Download the ImageNet2012 dataset

       train set：138GB     
       val set：6.3GB     
       ILSVRC2012_devkit_t12：2.45MB

    - Then, and move validation images to labeled subfolders, using ImageNet_val_split.py  need download imageNet val and ILSVRC2012_devkit_t12
    
     ```python
    python3.7 ImageNet_val_split.py ./val ./ILSVRC2012_devkit_t12
     ```
   ```
    move valimg to correspongding folders.
    official download the organize like:
    /val
        images
        images
       ......
    after the move the organize like:
    
    /val
       /n01440764
           images
       /n01443537
           images
       .....
    ```
## Training 

To train a model, run scripts with the desired model architecture and the path to the ImageNet dataset:

```bash
# 1p training 1p
bash test/train_full_1p.sh --data_path=imageNet_root_path

# 8p training 8p
bash test/train_full_8p.sh --data_path=imageNet_root_path

# To ONNX
python3.7 Efficient-B1_pth2onnx.py ./Efficient-b1.onnx

# eval default 8p， should support 1p
bash test/train_eval_8p.sh --data_path=imageNet_root_path

# test performer
bash test/train_performance_1p.sh --data_path=imageNet_root_path
bash test/train_performance_8p.sh --data_path=imageNet_root_path

# online inference demo 
python3.7.5 demo.py

```


## EfficientNet-B1 training result 

| Acc@1  | FPS  | Npu_nums | Epochs | AMP_Type |
| :----: | :--: | :------: | :----: | :------: |
|   -    | 451  |    1     |  100   |    O2    |
| 74.445 | 2073 |    8     |  100   |    O2    |

FPS = BatchSize * num_devices / time_avg

