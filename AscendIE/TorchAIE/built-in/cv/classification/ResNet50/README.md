# Resnet50

- [概述](#ZH-CN_TOPIC_0000001172161501)

- [环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)


- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Resnet是残差网络(Residual Network)的缩写,该系列网络广泛用于目标分类等领域以及作为计算机视觉任务主干经典神经网络的一部分，典型的网络有resnet50, resnet101等。Resnet网络的证明网络能够向更深（包含更多隐藏层）的方向发展。


- 参考实现：

  ```
  url=https://github.com/pytorch/examples/tree/main/imagenet
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 1 x 1000 | FLOAT32  | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下依赖

  **表 1**  版本配套表

| 配套                    | 版本              | 
|-----------------------|-----------------| 
| CANN                  | 6.3.RC2.alph002 | 链接                                                          |
| Python                | 3.9.0           |                                                           
| PyTorch               | 2.0.0           |
| torchVison            | 0.15.1          |-
| Ascend-cann-torch-aie |链接
| Ascend-cann-aie       |链接
| 芯片类型                  | Ascend310P3     | -                                                         |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>
## 安装CANN包

 ```
 chmod +x Ascend-cann-toolkit_6.3.RC2.alpha002_linux-aarch64.run 
./Ascend-cann-toolkit_6.3.RC2.alpha002_linux-aarch64.run --install
 ```
下载Ascend-cann-torch-aie和Ascend-cann-aie得到run包和压缩包
## 安装Ascend-cann-aie
 ```
  chmod +x Ascend-cann-aie_6.3.T200_linux-aarch64.run
  Ascend-cann-aie_6.3.T200_linux-aarch64.run --install
  cd Ascend-cann-aie
  source set_env.sh
  ```
## 安装Ascend-cann-torch-aie
 ```
 tar -zxvf Ascend-cann-torch-aie-6.3.T200-linux_aarch64.tar.gz
 pip3 install torch-aie-6.3.T200-linux_aarch64.whl
 ```

## 安装其他依赖
```
pip3 install pytorch==2.0.0
pip3 install torchVision==0.15.1
```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型使用ImageNet 50000张图片的验证集，请前往ImageNet官网下载数据集

    ```
    ├── ImageNet
    |   ├── val
    |   |    ├── ILSVRC2012_val_00000001.JPEG
    │   |    
    │   |    ├── ......
    |   ├── val_label.txt
    ```

2. 数据预处理。

   ```
   # 参考https://github.com/pytorch/examples/tree/main/imagnet/extract_ILSVRC.sh的处理。
   执行 valprep.sh脚本
   
   ```
## 模型推理<a name="section741711594517"></a>

1. 获取权重文件。

前往[Pytorch官方文档](https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet50)下载对应权重，参考下载权重如下：
   
      [权重](https://download.pytorch.org/models/resnet50-0676ba61.pth
2. 执行推理脚本
   ```
    python3 resnet50.py ./resnet50-0676ba61.pth
   ```


# 模型推理性能及精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用torch-aie推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度                                  | 性能 |
| --------- | ---------------- | ---------- |-------------------------------------| --------------- |
| 310P3 | 64 | ImageNet | top-1: 76.1624% <br>top-5: 92.8857% | 2580 |

