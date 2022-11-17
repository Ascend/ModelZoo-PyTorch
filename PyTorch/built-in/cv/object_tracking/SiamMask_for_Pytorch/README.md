# SiamMask-base for Pytorch
This folder contains the implementation of base training of the `SiamMask-base` on the `Youtube-VOS`, `COCO`, `ImageNet-DET` and `ImageNet-VID` datasets.

## Usage
### Install
- Install ASCEND-CANN, ASCEND-pytorch-1.8 and apex.
- Install other requirements from `requirements.txt` and `make.sh`.
  ```bash
  pip install -r requirements.txt
  bash make.sh
  ```

### Prepare Datasets
- We use `Youtube-VOS`, `COCO`, `ImageNet-DET` and `ImageNet-VID` datasets for training and `VOT2018` dataset for testing.
- It is recommended to symlink the dataset root to `$SiamMask-base/data`.
If your folder structure is different, you may need to change the corresponding paths in `$SiamMask-base/experiments/SiamMask-base_base/config.json`.
#### Training Datasets (ytb_vos + coco + det + vid)
Download and preprocess each datasets according the `readme.md` files in each folder at `$SiamMask-base/data/$Dataset_Name`.
#### Testing Dataset (VOT2018)
For Testing Dataset, run:
  ```bash
  cd $SiamMask-base/data
  apt install jq
  bash get_VOT2018_data.sh
  ```
### Prepare Pre-trained Model
For downloading the pre-trained model (174 MB), run:
  ```bash
  cd $SiamMask-base/models
  wget http://www.robots.ox.ac.uk/~qwang/resnet.model
  ```
(This model was trained on the ImageNet-1k Dataset.)

- After preparing, the file structure should look like:
  ```bash
  $ tree SiamMask_for_Pytorch
  SiamMask_for_Pytorch
  ├── data
  │   ├── ytb_vos
  │   │   ├── train.json
  │   │   └── crop511
  │   │       └── train
  │   │           ├── 05d77715782
  │   │           └── ...
  │   ├── coco
  │   │   ├── train2017.json
  │   │   └── crop511
  │   │       ├── train2017
  │   │       └── val2017
  │   ├── det
  │   │   ├── train2017.json
  │   │   └── crop511
  │   │       ├── ILSVRS2013_train
  │   │       ├── ILSVRC2014_train_0000          
  │   │       └── ...
  │   ├── vid
  │   │   ├── train.json
  │   │   └── crop511
  │   │       ├── ILSVRC2015_VID_train_0000
  │   │       ├── ILSVRC2015_VID_train_0001          
  │   │       └── ...
  │   ├── VOT2018
  │   │   ├── bag
  │   │   ├── ants3
  │   │   └── ...
  │   └── VOT2018.json
  ├── models
  │   ├── resnet.model
  │   └── ...
  ├── experiments
  ├── datasets
  └── ...
  
### Full Test (for accuracy)
For full test on 8 NPU, run:
```bash
cd $SiamMask-base/
bash ./test/train_full_8p.sh
```
Defaultly, training auto-resumes checkpoint in output directory. Remove the `output` directory to train from begin.

### Performance Test
For performance test on 1 NPU, run:

```bash
cd $SiamMask-base/
bash ./test/train_performance_1p.sh
```

For performance test on 8 NPU, run:

```bash
cd $SiamMask-base/
bash ./test/train_performance_8p.sh
```

### Training result for `SiamMask-base`

| SiamMask_Loss | FPS  | Npu_nums | Epochs | AMP_Type | CPU |
|:-------------:|:----:|:--------:|:------:|:--------:|:---:|
|       -       | 236  |    1     |   1    |    O1    | ARM |
|       -       | 1724 |    8     |   1    |    O1    | ARM |
|    2.5940     | 1615 |    8     |   20   |    O1    | ARM |

### FAQ
1. Why modify the learning rate and the weight in the configuration file  `$SiamMask-base/experiments/siammask_base/config.json`?
- This is a modification according to the description in `3.3 Implementation details` of the [paper](https://arxiv.org/abs/1812.05050)