# YOLACT

This implements training of YOLACT on the Microsoft COCO 2017 dataset, mainly modified from [dbolya/yolact](https://github.com/dbolya/yolact.git).

## YOLACT Detail

YOLACT is a simple, fully-convolutional model for real-time instance segmentation that achieves 29.8 mAP on MS COCO at 33.5 fps evaluated on a single Titan Xp, which is significantly faster than any previous competitive approach. Moreover, YOLACT obtain this result after training on only one GPU. 

YOLACT accomplish this by breaking instance segmentation into two parallel subtasks: (1) generating a set of prototype masks and (2) predicting per-instance mask coefficients. Then YOLACT produce instance masks by linearly combining the prototypes with the mask coefficients. Because this process doesn't depend on repooling, this approach produces very high-quality masks and exhibits temporal stability for free. Furthermore, the YOLACT paper analyze the emergent behavior of our prototypes and show they learn to localize instances on their own in a translation variant manner, despite being fully-convolutional. Finally, YOLACT also propose Fast NMS, a drop-in 12 ms faster replacement for standard NMS that only has a marginal performance penalty.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))

- `pip install -r requirements.txt`

- Get backbone pre-trained model.

    - ```
        cd YOLACT
        mkdir weights
        ```

    - Download `resnet101_reducedfc.pth` and put it into `./weights`.  

- Download the Microsoft coco 2017 dataset.

    - This is the directory of coco 2017 dataset.

      ```
      ├── coco
      	├── val2017/
      	├── train2017/
      	├── annotations/
      		├── instances_train2017.json
      		├── instances_val2017.json
      		├── ......
      ```

      

    - The recommended path to Microsoft coco 2017 dataset is  `/home/data/coco`

    

## Training

To train a model, run `train.py` with the desired model architecture and the path to the Microsoft coco 2017 dataset:

```bash
# training 1p accuracy, the first parameter is the path to training dataset
bash ./test/train_full_1p.sh /home/data/coco 

# training 1p performance
bash ./test/train_performance_1p.sh /home/data/coco

# training 8p accuracy
bash ./test/train_full_8p.sh /home/data/coco

# training 8p performance
bash ./test/train_performance_8p.sh /home/data/coco
```

Log path:
    test/output/devie_id/train_${device_id}.log           # training detail log
    test/output/0/YOLACT_bs8_8p_perf.log  # 8p training performance result log
    test/output/0/YOLACT_bs8_8p_acc.log   # 8p training accuracy result log



## YOLACT training result

| box mAP | mask mAP | FPS    | Npu_nums | Epochs   | Steps | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: | -------- | -------- |
| 31.98 | 29.62 | 25.4 | 8        | 54     | 100000 | O0      |

