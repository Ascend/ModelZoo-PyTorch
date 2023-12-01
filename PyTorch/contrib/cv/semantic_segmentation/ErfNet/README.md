# ErfNet

This implements training of ErfNet on the cityscapes dataset, mainly modified from [Eromera/erfnet_pytorch](https://github.com/Eromera/erfnet_pytorch).

## ErfNet Detail

- The original network structure contains ConvTranspose2d, but the operator can not converge in multi-p environment, so it is modified to interpolation and convolution.
- If you want to transfer learning based on the original weight, you only need to modify the fnum value of finetune script to the number of categories corresponding to your dataset.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))  

- Download the Cityscapes dataset from https://www.cityscapes-dataset.com/

  - Download the "leftImg8bit" for the RGB images and the "gtFine" for the labels.
  
  - Please note that for training you should use the "_labelTrainIds" and not the "_labelIds", you can download the [cityscapes scripts](https://github.com/mcordts/cityscapesScripts) and use the [conversor](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py) to generate trainIds from labelIds
  
# Expected dataset structure for [cityscapes](https://www.cityscapes-dataset.com/downloads/):
```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
    # below are generated Cityscapes panoptic annotation
    cityscapes_panoptic_train.json
    cityscapes_panoptic_train/
    cityscapes_panoptic_val.json
    cityscapes_panoptic_val/
    cityscapes_panoptic_test.json
    cityscapes_panoptic_test/
  leftImg8bit/
    train/
    val/
    test/
```
Install cityscapes scripts by:
```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

Note: to create labelTrainIds.png, first prepare the above structure, then run cityscapesescript with:
```
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```
These files are not needed for instance segmentation.

Note: to generate Cityscapes panoptic dataset, run cityscapesescript with:
```
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createPanopticImgs.py
```
These files are not needed for semantic and instance segmentation.
- Download the [erfnet_encoder_pretrained.pth.tar](https://github.com/Eromera/erfnet_pytorch/tree/master/trained_models/erfnet_encoder_pretrained.pth.tar) and put it into the `./trained_models` folder.
## Training

To train a model, run `train/main.py` with the desired model architecture and the path to the market dataset:

```bash
# training 1p performance
bash test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash test/train_performance_8p.sh --data_path=real_data_path

#test 8p accuracy
bash test/train_eval_8p.sh --data_path=real_data_path

# finetune
bash test/train_finetune_1p.sh --data_path=real_data_path

# Online inference demo
python demo.py
 
```

## ErfNet training result


| 名称      | iou      | fps      |
| :------: | :------: | :------:  |
| GPU-1p   | -        | 14.52      |
| GPU-8p   | -    | 94.64     |
| NPU-1p   | -        | 24.08      |
| NPU-8p   | 71.47    | 143.15     |



# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md