# Swin-Transformer
SSD: Single Shot MultiBox Detector | a PyTorch Model for Object Detection | VOC , COCO | Custom Object Detection  


### Dataset

* Pascal Visual Object Classes (VOC) data from the years 2007 and 2012.


## VOC dataset
VOC dataset contains images with twenty different types of objects.
```python
{'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'}
```


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
  python3 main.py --cfg configs/swin_t_64.yaml --local_rank 0 --data-path $data_path --batch-size 32
  ```
# Inference 
  ```
  python inference.py image_path checkpoint
  ```
 
