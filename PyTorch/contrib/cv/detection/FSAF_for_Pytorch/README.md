# FSAF_for_Pytorch

This project enables the FSAF model of the [mmdetection](https://github.com/open-mmlab/mmdetection) could be trained and evaluated on NPU, and remains the similar precision compared to the results of the GPU.

## FSAF Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.
Therefore, FSAF model need to be modified in the following aspects:

1. Converting tensors with the dynamic shapes into tensors with fixed shapes. (This is the hardest one)
2. Several operations, like the sum of `INT64`, are not supported on the NPU, so we modified tensors' `dtype` when needed, or put these operations on the CPU in some cases
3. Focal loss operation only supports GPUs, so we reimplemented the focal loss for the NPU
4. We used Apex for mmdtection due to the hardware defects of the NPU
5. ...

## Requirements

- NPU配套的run包安装（建议安装 20.2.0.rc1 版本，请用以下脚本确认版本号，不确保其他版本能正常训练/测评）

  ```sh
  ll /usr/local/Ascend/ascend-toolkit/latest
  ```

- Python v3.7.5
- PyTorch (NPU版本)
- Apex (NPU版本)
- MMCV v1.2.7
- MMDetection v2.10.0 (for training)
- MMDtection v2.11.0 (for exporting the ONNX model)

### Download and Prepare Dataset

1. Download this project and navigate to this path in your terminal
2. Download COCO dataset
3. Put your COCO dataset into this `FSAF/mmdetection/data` path

### Download and Modify MMCV

1. Clone MMCV v1.2.7 into `FSAF` path

    ```sh
    git clone -b v1.2.7 git://github.com/open-mmlab/mmcv.git
    ```

2. Then your file structure should be like the following graph, use the `mmcv_need` to replace `the source code of mmcv` by the following code.

    ```plain
    - mmcv (this is what you clone, we call it the root path of mmcv)
      - mmcv (this is the source code of mmcv)
      - ...
    - mmcv_need
    - mmdetection
    - scripts
    - README.md
    ```

    ```shell
    cd FSAF
    rm -rf mmcv/mmcv
    cp -r mmcv_need mmcv
    mv mmcv/mmcv_need mmcv/mmcv
    ```

### Configure the Environment

1. Navigate into the `root path of the mmcv`, and build MMCV

    ```sh
    cd mmcv
    export MMCV_WITH_OPS=1
    export MAX_JOBS=8
    python3.7.5 setup.py build_ext
    python3.7.5 setup.py develop
    pip3.7.5 list | grep mmcv
    ```

2. Installing MMDetection

    ```sh
    cd mmdetection
    pip3.7.5 install -r requirements/build.txt
    python3.7.5 setup.py develop
    pip3.7.5 list | grep mmdet
    ```

3. Change the following code in Apex for `O1` opt-level supporting on NPU, you can find this code in `{the path of the fsaf environment in conda}/lib/python3.7/site-packages/apex/amp/utils.py`, mine is `/root/archiconda3/envs/fsaf/lib/python3.7/site-packages/apex/amp/utils.py`

    ```diff
    # change this line (line 113)
    - if cached_x.grad_fn.next_functions[1][0].variable is not x:
    # into this
    + if cached_x.grad_fn.next_functions[0][0].variable is not x:
    ```

## Training

> The training outputs directory is in `mmdetection/work_dirs` by default. You can find models and logs in this directory.

### Training on Single NPU

```sh
cd FSAF
chmod +x ./test/train_full_1p.sh
cd mmdetection
bash ../test/train_full_1p.sh --data_path=./mmdetection/data/coco
```

### Training on NPUs

```sh
cd FSAF
chmod +x ./test/train_full_8p.sh
chmod +x ./mmdetection/tools/dist_train.sh
cd mmdetection
bash ../test/train_full_8p.sh --data_path=./mmdetection/data/coco
```

### Evaluation

```sh
cd FSAF
chmod +x ./test/train_eval_8p.sh
chmod +x ./mmdetection/tools/dist_test.sh
cd mmdetection
bash ../test/train_eval_8p.sh {你想测试的模型路径，可不传，默认为./work_dirs/fsaf_r50_fpn_1x_coco/latest.pth} 
```

### Demo

```sh
cd FSAF 
source ./test/env_npu.sh
python3.7.5 demo.py {the path of your FSAF checkpoint，可不传，默认为./mmdetection/work_dirs/fsaf_r50_fpn_1x_coco/latest.pth}
```

### Exporting to the ONNX

We need to clone the MMDetection v2.11.0 for exporting FSAF checkpoint into the ONNX model due of some bugs in MMDetection v2.10.0

> 本次训练功能在 MMDetection v2.10 版本的基础上打通，经过测试，在 2.10 版本训练完成的 FSAF 权重文件可在 MMDetection v2.11.0 上正常推理，MMDetection v2.11.0 以上版本请自测。

```sh
cd FSAF
source ./test/env_npu.sh
git clone https://github.com/open-mmlab/mmdetection.git mmdetection-2.11.0
cd mmdetection-2.11.0
git checkout 2894516 # 切换到 v2.11.0 版本
python3.7.5 setup.py develop
```

```sh
# back to the path of the FSAF
cd FSAF
source ./test/env_npu.sh
python3.7.5 pytorch2onnx.py {config of FSAF，可不传，默认为./mmdetection/configs/fsaf/fsaf_r50_fpn_1x_coco.py} {the path of your FSAF checkpoint，可不传，默认为./mmdetection/work_dirs/fsaf_r50_fpn_1x_coco/latest.pth}
```

## FSAF Training Results

| Acc@1    | FPS       | # of NPU/GPU | Epochs   | Opt-Level | Loss Scale |
| :------: | :------:  | :------:     | :------: | :------:  | :------:   |
| 36.0     | -         | 8P GPU       | 12       | FP32      | -          |
| 9.4      | 11.39     | 1P GPU       | 1        | O1        | dynamic    |
| 37.5     | 70.84     | 8P GPU       | 12       | O1        | dynamic    |
| 10.3     | 1.26      | 1P NPU       | 1        | O1        | 32.0       |
| 36.2     | 8.38      | 8P NPU       | 12       | O1        | 32.0       |
