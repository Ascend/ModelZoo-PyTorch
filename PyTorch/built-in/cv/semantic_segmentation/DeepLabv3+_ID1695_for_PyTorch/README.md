# Deeplabv3+

Deeplabv3+ is an end-2-end semantic segmentation model. This implements training of deeplabv3+ on the PASCAL VOC2012 dataset, mainly modified from (https://github.com/jfzhang95/pytorch-deeplab-xception).


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install matplotlib pillow tensorboardX tqdm`
- modify env.sh to match your server setttings
- Model is trained and tested against PASCAL VOC2012 dataset, if you already have the dataset, change line 6 in ./mypath.py to point to '$VOCdevkit/VOC2012'
  if you dont have the data installed on server, download and prepare like this(then change the path to ./data/VOC2012): 
```
   mkdir data
   cd data
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
   tar -xf VOCtrainval_11-May-2012.tar
```
## Training
Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision.
Suggestion: the pillow is 9.1.0 and the torchvision is 0.6.0

To train a model, run "train_NPU_fp32.py" or "train_NPU_fp32_8p.py"with the desired model architecture and the other hyper-parameters.
For testing, use "train_NPU.py" or "train_NPU_8p.py" (have to modify env.sh).

You can also directly use the scripts for training:

```
bash ./test/train_performance_1p.sh --data_path=`data path`  --epochs='train epochs'  # 1-board training for performance
bash ./test/train_performance_8p.sh --data_path=`data path`  --epochs='train epochs'   # 8-board training for performance

bash ./test/train_full_1p.sh --data_path=`data path` --epochs='train epochs'          # 1-board training for accuracy
bash ./test/train_full_8p.sh --data_path=`data path` --epochs='train epochs'          # 8-board training for accuracy

```

## Deeplabv3+ training result

| mIOU    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------:| :------: | :------:  | :------: | :------: |
| 0.759   | 13.96     | 1        | 100      | fp32     |
| 0.751   | 65.86     | 8        | 100      | fp32     |
