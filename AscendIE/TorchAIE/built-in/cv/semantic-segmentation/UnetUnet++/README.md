
# UNet/UNet++模型推理

- [概述](#ZH-CN_TOPIC_0000001172161501)

   - [输入输出数据](#ZH-CN_TOPIC_0000001126281702)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [安装依赖](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理精度](#ZH-CN_TOPIC_0000001172201573)

  ******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

语义分割(Semantic Segmentation)是图像处理和机器视觉一个重要分支。与分类任务不同，语义分割需要判断图像每个像素点的类别，进行精确分割。语义分割目前在自动驾驶、自动抠图、医疗影像等领域有着比较广泛的应用。

UNet/UNet++是在医学图像处理领域应用广泛的语义分割网络。本文旨在提供基于推理引擎的UNet/UNet++参考样例，使用了Kaggle's Carvana Image Masking Challenge数据集，并测试了昇腾310P芯片上的推理精度供参考，模型性能仍在持续优化中。


- 参考实现：

  ```shell
  UNet(https://github.com/milesial/Pytorch-UNet)
  branch=master
  UNet++(https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets)
  branch=master
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | images   | RGB_FP32 | batchsize x 3 x 512 x 512 | NCHW         |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下依赖

  **表 1**  版本配套表

| 配套                    | 版本              |
|-----------------------|-----------------|
| CANN                  | 7.0.T3 | -                                                       |
| Python                | 3.9         |
| PyTorch               | 2.0.1           |
| torchVison            | 0.15.2          |-
| Ascend-cann-torch-aie | 7.0.T3
| Ascend-cann-aie       | 7.0.T3
| 芯片类型                  | Ascend310P3     | -                                                         |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>
## 安装CANN包
### Arm安装
 ```
 chmod +x Ascend-cann-toolkit_7.0.T3_linux-aarch64.run
./Ascend-cann-toolkit_7.0.T3_linux-aarch64.run --install
 ```
### X84安装
 ```
 chmod +x Ascend-cann-toolkit_7.0.T3_linux-x86_64.run
./Ascend-cann-toolkit_7.0.T3_linux-x86_64.run --install
 ```
## 安装Ascend-cann-aie
1. 安装
```
chmod +x ./Ascend-cann-aie_${version}_linux-${arch}.run
./Ascend-cann-aie_${version}_linux-${arch}.run --check
# 默认路径安装
./Ascend-cann-aie_${version}_linux-${arch}.run --install
# 指定路径安装
./Ascend-cann-aie_${version}_linux-${arch}.run --install-path=${AieInstallPath}
```
2. 设置环境变量
```
cd ${AieInstallPath}
source set_env.sh
```
## 安装Ascend-cann-torch-aie
1. 安装
 ```
# 安装依赖
conda create -n py39_pt2.0 python=3.9.0 -c pytorch -y
conda install decorator -y
pip install attrs
pip install scipy
pip install synr==0.5.0
pip install tornado
pip install psutil
pip install cloudpickle
wget https://download.pytorch.org/whl/cpu/torch-2.0.1%2Bcpu-cp39-cp39-linux_x86_64.whl
pip install torch-2.0.1+cpu-cp39-cp39-linux_x86_64.whl

# 解压
tar -xvf Ascend-cann-torch-aie-${version}-linux-${arch}.tar.gz
cd Ascend-cann-torch-aie-${version}-linux-${arch}

# C++运行模式
chmod +x Ascend-cann-torch-aie_${version}_linux-${arch}.run
# 默认路径安装
./Ascend-cann-torch-aie_${version}_linux-${arch}.run --install
# 指定路径安装
./Ascend-cann-torch-aie_${version}_linux-${arch}.run --install-path=${TorchAIEInstallPath}

# python运行模式
pip install torch_aie-${version}-cp{pyVersion}-linux_x86_64.whl
 ```
 > 说明：如果用户环境是[libtorch1.11](https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-1.11.0%2Bcu113.zip)，需要使用配套的torch 1.11-cpu生成yolov5的torchscript，再配套使用torch-aie-torch1.11的包。

2. 设置环境变量
```
cd ${TorchAIEInstallPath}
source set_env.sh
```

3. 卸载torch aie
```
pip uninstall torch-aie
cd ${TorchAIEInstallPath}/latest/scripts
bash uninstall.sh
```

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 安装依赖<a name="section4622531142816"></a>

   ```
   pip install numpy==1.23
   pip install tqdm
   pip install opencv-python
   pip install pandas==2.0.2
   pip install requests
   pip install pyyaml
   pip install Pillow==9.5
   wget https://download.pytorch.org/whl/cpu/torchvision-0.15.2%2Bcpu-cp39-cp39-linux_x86_64.whl
   pip install torchvision-0.15.2+cpu-cp39-cp39-linux_x86_64.whl
   pip install matplotlib
   pip install seaborn
   ```

## 准备数据集<a name="section183221994411"></a>

本模型需要Kaggle's Carvana Image Masking Challenge数据集，下载[地址](https://www.kaggle.com/c/carvana-image-masking-challenge)

   数据集结构如下
   ```
    test
     |-- images
     |   |-- fecea3036c59_01.jpg
     |   |-- fecea3036c59_02.jpg
     |   |-- ...
     `-- masks
         |-- fecea3036c59_01_mask.gif
         |-- fecea3036c59_02_mask.gif
         `-- ...
   ```


## 模型推理<a name="section741711594517"></a>
1. 获取权重文件。
```
(U-Net) https://github.com/milesial/Pytorch-UNet/releases/tag/v3.0
(U-Net++) 使用随机权重
```
2. 执行python脚本
    ```
    python3 sample.py --image_path path_to_image --mask_path path_to_mask --model_name "unet" --pth path_to_weight --batch_size 1 --device 0 --loop 100 --warm_counter 10
    ```



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

Unet模型精度与性能如下表

| 芯片型号 | Batch Size   | 数据集 | 精度(dice score) | 性能(fps)|
| --------- | ---------------- | ---------- | ---------- | --------------- |
|    Ascend310P3       |       1    |   Carvana Image Masking Challenge     |      97.4%      |       50.968         |


Unet++模型精度与性能如下表

| 芯片型号 | Batch Size   | 数据集 | 精度(cosine similarity with torch) | 性能(fps) |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|    Ascend310P3       |       1    |   --------         |     100.0%      |   25.008             |
