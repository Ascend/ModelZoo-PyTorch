# SSD_MobileNet
SSD: Single Shot MultiBox Detector | a PyTorch Model for Object Detection | VOC , COCO | Custom Object Detection  

This repo contains code for [Single Shot Multibox Detector (SSD)](https://arxiv.org/abs/1512.02325) with custom backbone networks. The authors' original implementation can be found [here](https://github.com/weiliu89/caffe/tree/ssd).

### Dataset

* Pascal Visual Object Classes (VOC) data from the years 2007 and 2012.
* COCO.
* Custom Dataset.

## VOC dataset
VOC dataset contains images with twenty different types of objects.
```python
{'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'}
```

Each image can contain one or more ground truth objects.

Each object is represented by –

- a bounding box in absolute boundary coordinates

- a label (one of the object types mentioned above)

-  a perceived detection difficulty (either `0`, meaning _not difficult_, or `1`, meaning _difficult_)

### Download

Specfically, you will need to download the following VOC datasets –

- [2007 _trainval_](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) (460MB)

- [2012 _trainval_](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) (2GB)

- [2007 _test_](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) (451MB)


### Inputs to model

We will need three inputs.

#### Images

* For SSD300 variant, the images would need to be sized at `300, 300` pixels and in the RGB format.
* PyTorch follows the NCHW convention, which means the channels dimension (C) must precede the size dimensions(1, 3, 300, 300).

Therefore, **images fed to the model must be a `Float` tensor of dimensions `N, 3, 300, 300`**, and must be normalized by the aforesaid mean and standard deviation. `N` is the batch size.


#### Objects' Bounding Boxes

For each image, the bounding boxes of the ground truth objects follows (x_min, y_min, x_max, y_max) format`.

# Training
* In config.json change the paths. 
* "backbone_network" : "MobileNetV2" or "MobileNetV1"
* For training run
  ```
  python train.py config.json
  ```
# Inference 
  ```
  python inference.py image_path checkpoint
  ```
 
