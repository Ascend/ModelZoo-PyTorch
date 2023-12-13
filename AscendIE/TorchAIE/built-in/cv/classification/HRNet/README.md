# HRNet

- [概述](#ZH-CN_TOPIC_0000001172161501)

- [环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

- [模型推理精度](#ZH-CN_TOPIC_0000001172201573)

  ******



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

高解析网络（HRNet）在整个过程中保持高解析呈现。该网络有两个关键特性：1. 并行连接高到低分辨率的卷积流；2. 在决议之间反复交换信息。优点是得到的呈现在语义上更丰富，空间上更精确。HRNet在广泛的应用中具有优越性，包括人体姿态估计、语义分割和对象检测，表明HRNet是解决计算机视觉问题强有力的工具。

- 参考论文：[Deep High-Resolution Representation Learning for Visual Recognition](https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fpdf%2F1908.07919.pdf)

- 参考实现：

```
url=https://github.com/HRNet/HRNet-Image-Classification.git
branch=master
model_name=HRNet-W18-C
```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

      | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
      | -------- | -------- | ------------------------- | ------------ |
      | input    | FLOAT32  | batchsize x 3 x 224 x 224 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小                  | 数据排布格式 |
  | -------- | -------- | --------              | ------------ |
  | output   | FLOAT32  | batchsize x num_class | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下依赖

  **表 1**  版本配套表

| 配套                    | 版本              | 
|-----------------------|-----------------| 
| 固件与驱动                 | 23.0.rc1                |
| CANN                  | 	7.0.RC1.alpha003| -                                                       |
| Python                | 	3.9.11           |                                                           
| PyTorch               | 2.0.1           |
| torchVison            | 0.15.2          |-
| timm                  | 0.6.13          | -                                                         |
| Torch_AIE                  | 6.3.rc2              | -                                                         |
| 芯片类型                  | Ascend310P3     | -                                                         |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/HRNet/HRNet-Image-Classification.git
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

0. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型使用ImageNet 50000张图片的验证集，请前往[ImageNet官网](https://image-net.org/download.php)下载数据集
    ```
    ├── ImageNet
    |   ├── val
    |   |    ├── ILSVRC2012_val_00000001.JPEG
    │   |    
    │   |    ├── ......
    |   ├── val_label.txt
    ```


1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.pt文件。

   1. 获取权重文件。并将权重文件放在当前目录下。

       下载链接：

		 [oneDrive](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw)

       [pan.baidu.com (AccessCode:r5xn)](https://pan.baidu.com/s/1Px_g1E2BLVRkKC5t-b-R5Q)

   2. 导出torchscript模型文件。

      1. 使用export.py导出torchscript文件。

         运行export.py脚本。

         ```
          python3 export.py --cfg ./HRNet-Image-Classification/experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml --model_path ./hrnetv2_w18_imagenet_pretrained.pth --ts_save_path hrnet.pt
         ```

         - 参数说明：

            -   --cfg：模型配置。
            -   --model_path：权重文件。
            -   --ts_save_path：torchscript模型文件。

         获得hrnet.pt文件。

2. 运行模型评估脚本，测试ImageNet验证集推理精度
 ```
    python3 eval.py --batch_size 1 --device_id 0 --dataset /path/to/val
        --label_file /path/to/val_label.txt --traced_model hrnet.pt
 ```

# 模型推理性能及精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用torch-aie推理计算，精度参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度                                | 性能       |
| --------- |------------| ---------- |-----------------------------------|----------|
| 310P3 | 1          | ImageNet | top-1: 76.756% <br>top-5: 93.446% | 500 FPS  |
| 310P3 | 4          | ImageNet | top-1: 76.756% <br>top-5: 93.446% | 1333 FPS |
| 310P3 | 8          | ImageNet | top-1: 76.756% <br>top-5: 93.446% | 1305 FPS |
| 310P3 | 16         | ImageNet | top-1: 76.756% <br>top-5: 93.446% | 1511 FPS |
| 310P3 | 32         | ImageNet | top-1: 76.756% <br>top-5: 93.446% | 1562 FPS |
| 310P3 | 64         | ImageNet | top-1: 76.756% <br>top-5: 93.446% | 1315 FPS |