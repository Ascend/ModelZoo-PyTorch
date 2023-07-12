# Hourglass

This project enables the Hourglass model of the [mmpose](https://github.com/open-mmlab/mmpose) could be trained and evaluated on NPU, and remains the similar precision compared to the results of the GPU.

## Hourglass Detail

Hourglass model has been modified in the following aspects:

1. change the device from gpu to npu for the model
2. change the input data from gpu to npu
4. Used Apex for Hourglass model due to the hardware defects of the NPU

## Requirements

- NPU supporting run package(it is recommended to install version 20.2.rc1, please use the following command to confirm the version of the installed run package, we do not ensure that other versions can be used for normal training and evaluation for the Hourglass Model)

  ```
  ll /usr/local/Ascend/ascend-toolkit/latest
  ```

- Python v3.7.5

- PyTorch (ascend version)

- Apex (ascend version)

- MMCV v1.2.7

- mmpose v0.8.0 

### Download and prepare dataset

1. This Hourglass model need MPII dataset,so  downloading MPII dataset is needed，you can download MPII dataset from http://human-pose.mpi-inf.mpg.de/#download

2. Because open-mmlab has converted the original annotation files into json format, so, please download them from https://download.openmmlab.com/mmpose/datasets/mpii_annotations.tar

3. Extract and put your MPII dataset and annotations files into this  path:`Hourglass/mmpose-master/data` .The struct of the data file should be like this:

   ```python
   |--data
   	|--mpii
     		|--annotations
           	|--mpii_gt_val.mat
               |--mpii_test.json
               |--mpii_train.json
               |--mpii_trainval.json
               |--mpii_val.json
      		|--images
           	|--000001163.jpg
               |--000003072.jpg
               |--xxxxxxxxx.jpg
   ```


### Build and install MMCV

1. You should use the following command to download mmcv in version 1.2.7.

    ```shell
    git clone -b v1.2.7 git://github.com/open-mmlab/mmcv.git
    ```

2. You should navigate(cd) into the path:`Hourglass`, and then, use the following commands to replace some files for the mmcv which you download.

    **state:** {your mmcv file path} should be change to the real path of  the mmcv file which you have been downloaded.

    ```shell
    cp -f mmcv_need/builder.py {your mmcv file path}/mmcv/runner/optimizer/
    cp -f mmcv_need/distributed.py {your mmcv file path}/mmcv/parallel/
    cp -f mmcv_need/data_parallel.py {your mmcv file path}/mmcv/parallel/
    cp -f mmcv_need/dist_utils.py {your mmcv file path}/mmcv/runner/
    cp -f mmcv_need/optimizer.py {your mmcv file path}/mmcv/runner/hooks/
    cp -f mmcv_need/checkpoint.py {your mmcv file path}mmcv/runner/
    cp -f mmcv_need/iter_timer.py {your mmcv file path}/mmcv/runner/hooks/
    cp -f mmcv_need/base_runner.py {your mmcv file path}mmcv/runner/
    cp -f mmcv_need/epoch_based_runner.py {your mmcv file path}/mmcv/runner/
    ```

3. You should navigate(cd) into the path of your mmcv file. And then, you should use the following commands to build and install the mmcv.

    ```sh
    export MMCV_WITH_OPS=1
    export MAX_JOBS=8
    python setup.py build_ext
    python setup.py develop
    ```

### Install mmpose

You should navigate(cd) into the path of the mmpose file:`Hourglass/mmpose-master` . And then, you should use the following commands to install the mmpose.

```sh
pip install -r requirements.txt
python setup.py develop
```

### Change code of Apex

Change the following code in Apex for supporting on NPU, you can find this code in the path:`{the path of your conda enviroment}/lib/python3.7/site-packages/apex/amp/utils.py`.

For example:

`/root/archiconda3/envs/Hourglass/lib/python3.7/site-packages/apex/amp/utils.py`

```diff
# change this line (line 113)
- if cached_x.grad_fn.next_functions[1][0].variable is not x:
# into this
+ if cached_x.grad_fn.next_functions[0][0].variable is not x:
```

## Training

> The training outputs directory is in `Hourglass/mmpose-master/work_dirs` by default. You can find models and logs in this directory.

### Source your environment variables

You should navigate(cd) into the path of the `Hourglass`, and then, you can use following command to source your environment variables. 

```sh
source env.sh
```

### Training model on single NPU

You should navigate(cd) into the path of the scripts file:`Hourglass`, and then, you can use following command to training model on single NPU. 

```sh
bash scripts/train_1p.sh
```

### Training on 8P NPUs

You should navigate(cd) into the path of the scripts file:`Hourglass`, and then, you can use following command to training model on 8P NPUs.

```sh
bash scripts/train_8p.sh
```

### Evaluation

You should navigate(cd) into the path of the scripts file:`Hourglass`, and then, you can use following command to eval.

```sh
bash scripts/eval.sh [number of npus]
```

**State:**

[number of npus] : This is an optional parameter, the default value is 8.

**Example:**

```shell
bash scripts/eval.sh
```

​																	or

```shell
bash scripts/eval.sh 1
```

### Demo

You can use following command to run the demo.py script. And after the script successful run complete, you can find the original image and the image of inference result in this file path: Hourglass/mmpose-master/demo/mpii_demo_img/demo_mpii.jpg ( and infer_demo_mpii.jpg )

```
python3 demo.py [--config {the path of your config}] [--checkpoint {the path of your checkpoint}]
```

**state**

[--config {the path of your config}] and [--checkpoint {the path of your checkpoint}] are both optional parameters, you can set your config file and checkpoint file by these parameters.

**Example:**

```
python3 demo.py
```

​																					or

```
python3 demo.py --config ./mmpose-master/configs/top_down/hourglass/mpii/hourglass52_mpii_384x384.py --checkpoint ./mmpose-master/work_dirs/hourglass52_mpii_384x384/latest.pth
```

### Exporting to the ONNX

1.  First, You need to install onnx and onnxruntime using following command.

   ```shell
   pip install onnx onnxruntime
   ```

2. Then, You should navigate(cd) into the path of the Hourglass file:`Hourglass`, and then,you can use following command to export checkpoint into the ONNX model.And after the script successful run complete, you can find the onnx file named `hourglass.onnx` in the same file.

   ```
   python3 pytorch2onnx.py [--config {the path of your config}] [--checkpoint {the path of your checkpoint}]
   ```

   **State:**

   [--config {the path of your config}] and [--checkpoint {the path of your checkpoint}] are both optional parameters, you can set your config file and checkpoint file by these parameters.

   **Example:**

   ```
   python3 pytorch2onnx.py
   ```
   
   ​																			  or
   
   ```
   python3 pytorch2onnx.py --config ./mmpose-master/configs/top_down/hourglass/mpii/hourglass52_mpii_384x384.py --checkpoint ./mmpose-master/work_dirs/hourglass52_mpii_384x384/latest.pth
   ```

## Hourglass Training Results

|  PCKh   |  FPS   | # of NPU/GPU | Run Epochs | Opt-Level | Loss Scale |
| :-----: | :----: | :----------: | :--------: | :-------: | :--------: |
|    -    | 45.85  |    1P GPU    |     1      |    O2     |  dynamic   |
| 89.0189 | 286.41 |    8P GPU    |    210     |    O2     |  dynamic   |
|    -    | 20.16  |    1P NPU    |     1      |    O2     |  32768.0   |
| 88.9383 | 57.344 |    8P NPU    |    210     |    O2     |  32768.0   |


# Statement
For details about the public address of the code in this repository, you can get from the file public_address_statement.md
