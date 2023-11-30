# 3D human pose estimation in video with temporal convolutions
This implementation mainly trains the VideoPose3D in a supervised manner, modified from https://github.com/facebookresearch/VideoPose3D

## VideoPose3D Details
Due to the Ascend-Python's low-efficient implementation on Conv1D, this version changes all conv1d to conv2d.


## Results on Human3.6M
Under Protocol 1 (mean per-joint position error).

| results | FPS | epoches | AMP_Type | Device(s) |
|:-------|:-------:|:-------:|:-------:|:--------:|
| 46.5mm | 5544 | 80 | O1 | 1p NPU |
| 46.79 mm | 40757 | 80 | O1 | 8p NPU | 

## Quick start
To get started as quickly as possible, follow the instructions in this section. This should allow you train a model from scratch, test our pretrained models, and produce basic visualizations. For more detailed instructions, please refer to [`DOCUMENTATION.md`](DOCUMENTATION.md).

### Dependencies
Make sure you have the following dependencies installed before proceeding:
- Python 3+ distribution
- PyTorch 1.5.0
- `pip install -r requirements.txt`


### Dataset setup
In order to proceed, you must also copy the 'data' file we provided into the VideoPose3D root directory.

## Train

```
# 1p train perf
bash test/train_performance_1p.sh

# 8p train perf
bash test/train_performance_8p.sh

# 1p train full
bash test/train_full_1p.sh

# 8p train full
bash test/train_full_8p.sh

# 8p evaluate full
bash test/eval_full_8p.sh
```


## License
This work is licensed under CC BY-NC. See LICENSE for details. Third-party datasets are subject to their respective licenses.
If you use our code/models in your research, please cite our paper:
```
@inproceedings{pavllo:videopose3d:2019,
  title={3D human pose estimation in video with temporal convolutions and semi-supervised training},
  author={Pavllo, Dario and Feichtenhofer, Christoph and Grangier, David and Auli, Michael},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md