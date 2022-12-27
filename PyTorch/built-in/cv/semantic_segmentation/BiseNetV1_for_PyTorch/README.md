# BiseNetV1 for Pytorch
This folder contains the implementation of training of the `BiseNetV1` on the `CityScapes` datasets based on `mmsegmentation` framework.

## Usage
### Install
#### NPU-related components
- Install ASCEND-CANN, ASCEND-pytorch-1.8 and apex.
- Install `torchvison 0.9.1` and `pillow 9.3.0`.
  ```bash
  cd $BiseNetV1_for_PyTorch
  pip install torchvision==0.9.1
  pip install pillow==9.3.0
  ```

#### mmcv-full
- Download `mmcv-full` v1.6.1 from github:
  ```bash
  cd $BiseNetV1_for_PyTorch
  git clone -b v1.6.1 --depth=1 https://github.com/open-mmlab/mmcv.git
  ```

- Replace the MMCV file to adapt to the NPU:
  ```bash
  cd $BiseNetV1_for_PyTorch
  /bin/cp -f mmcv_need/builder.py ${mmcv_path}/mmcv/runner/optimizer/
  /bin/cp -f mmcv_need/dist_utils.py ${mmcv_path}/mmcv/runner/
  /bin/cp -f mmcv_need/__init__utils.py ${mmcv_path}/mmcv/utils/__init__.py
  /bin/cp -f mmcv_need/device_type.py ${mmcv_path}/mmcv/utils/
  /bin/cp -f mmcv_need/optimizer.py ${mmcv_path}/mmcv/runner/hooks/
  /bin/cp -f mmcv_need/__init__device.py ${mmcv_path}/mmcv/device/__init__.py
  /bin/cp -rf mmcv_need/npu ${mmcv_path}/mmcv/device/
  /bin/cp -f mmcv_need/utils.py ${mmcv_path}/mmcv/device/
  ```

- Compiling and installing the MMCV (10 mins):
  ```bash
  cd $BiseNetV1_for_PyTorch/mmcv
  pip install -r requirements/optional.txt
  MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
  ```
  
#### mmsegmentation
- To install mmesegmentation, run:
  ```bash
  cd $BiseNetV1_for_PyTorch
  pip install -v -e .
  # "-v" means verbose, or more output
  # "-e" means installing a project in editable mode,
  # thus any local modifications made to the code will take effect without reinstallation.
  ```

### Prepare Datasets
- The data can be found [here](https://www.cityscapes-dataset.com/downloads/) after registration. Download `gtFine_trainvaltest.zip` (241MB) and `leftImg8bit_trainvaltest.zip` (11GB). 
- The compressed package should be stored in `$BiseNetV1_for_PyTorch/data`.
- To decompress the dataset, run:
  ```shell
  cd $BiseNetV1_for_PyTorch/data
  apt install unzip
  unzip leftImg8bit_trainvaltest.zip -d cityscapes
  # when prompting whether to overwrite, enter 'A'
  unzip gtFine_trainvaltest.zip -d cityscapes
  ```
- After decompression, the file structure should look like:
  ```none
  $ BiseNetV1_for_PyTorch
    ├── mmseg
    ├── tools
    ├── configs
    ├── data
    │   └── cityscapes
    │       ├── leftImg8bit
    │       │   ├── train
    │       │   ├── val
    │       │   └── test
    │       ├── gtFine
    │       │   ├── train
    │       │   ├── val
    │       │   └── test
    │       ├── train.txt
    │       ├── val.txt
    │       └── test.txt
    └── ... 
  ```

- By convention, `**labelTrainIds.png` are used for cityscapes training. To generate `**labelTrainIds.png`, run:
  ```shell
  cd $BiseNetV1_for_PyTorch
  pip install cityscapesscripts
  # --nproc means 8 process for conversion, which could be omitted as well.
  python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
  ```

### Full Test (for accuracy)
For full test on 8 NPU, run:
```bash
cd $BiseNetV1_for_PyTorch
bash ./test/train_full_8p.sh
```
Defaultly, training auto-resumes checkpoint in work directory. Remove the `work_dirs` directory to train from begin.

### Performance Test
For performance test on 1 NPU, run:

```bash
cd $BiseNetV1_for_PyTorch
bash ./test/train_performance_1p.sh
```

For performance test on 8 NPU, run:

```bash
cd $BiseNetV1_for_PyTorch
bash ./test/train_performance_8p.sh
```

### Training result for `BiseNetV1`

| mIoU | FPS | Npu_nums | Steps | AMP_Type | CPU |
|:----:|:---:|:--------:|:-----:|:--------:|:---:|
|  -   |  -  |    1     |  400  |    O1    | ARM |
|  -   |  -  |    8     |  400  |    O1    | ARM |
|  -   |  -  |    8     | 40000 |    O1    | ARM |

### Notes
It is recommended to use `python` or `python3.7` to execute the model training process. If you need to use `python3`, run the following commands before using `python3` due to the `2to3` dependency.
```bash
unlink /usr/bin/python3
ln -s /usr/bin/python3.7 /usr/bin/python3
```