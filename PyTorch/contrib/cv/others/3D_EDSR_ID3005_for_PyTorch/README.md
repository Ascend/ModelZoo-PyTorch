# 3D_EDSR

The 3D EDSR (Enhanced Deep Super Resolution) convolution neural network used in this repository is based on the implementation of the CVPR2017 workshop Paper: "Enhanced Deep Residual Networks for Single Image Super-Resolution" (https://arxiv.org/pdf/1707.02921.pdf) using PyTorch on the X-Ray Micro-CT images, mainly modified from [GitHub - sci-sjj/EDSRmodelling](https://github.com/sci-sjj/EDSRmodelling).
Q
## 3D_EDSR Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.Therefore, 3D_EDSR is re-implemented using semantics such as custom OP.

## Requirements

- Install PyTorch ([pytorch.org](https://gitee.com/link?target=http%3A%2F%2Fpytorch.org))
- `pip install -r requirements.txt`
- Download the Multi-resolution X-Ray micro-CT images of Bentheimer Sandstones dataset from [Zenodo at DOI: 10.5281/zenodo.5542624](https://doi.org/10.5281/zenodo.5542623).

##  training/testing dataset

To generate suitable training/testing images (sub-slices of the full data above), the following code can be run:

- dataset_generator.py. This generates LR and registered x3 HR sub-images for 3D EDSR training/testing. The LR/HR sub-images are separated into two different folders LR and HR

## Training

The 3D EDSR model can then be trained on the LR and HR sub-sampled data via:

- main_edsr.py. This trains the 3D EDSR network on the LR/HR data. It requires the code load_data.py, which is the sub-image loader for 3D EDSR training. It also requires the 3D EDSR model structure code edsr_x3_3d.py. The code then saves the trained network as 3D_EDSR.pt.

```shell
# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=real_data_path

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path
```

Log path: test/output/devie_id/train_${device_id}.log or obs://cann-idxxx/npu/workspace/MA-new-npu-xxx/log/modelarts-job-xxx-worker-0.log

## 3D_EDSR result

|           | PSNR (dB) | Npu_nums | Epochs | AMP_Type | FPS     |single step cost|
| --------- | --------- | -------- | ------ | -------- |-------- |--------|
| NPU       | 23.0884   | 1        | 50     | O2       | 1.12    | 0.43 |
|           | PSNR (dB) | Gpu_nums | Epochs | AMP_Type | FPS     |single step cost|
| GPU       | 23.0939   | 1        | 50     | O2       | 4.54    | 0.65|



