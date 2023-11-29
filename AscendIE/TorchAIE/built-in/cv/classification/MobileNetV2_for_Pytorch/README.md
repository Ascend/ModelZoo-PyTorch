# MobileNetV2-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

- [环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

- [模型推理精度](#ZH-CN_TOPIC_0000001172201573)

  ******

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

mobileNetV2是对mobileNetV1的改进，是一种轻量级的神经网络。mobileNetV2保留了V1版本的深度可分离卷积，增加了线性瓶颈（Linear Bottleneck）和倒残差（Inverted Residual）。

- 参考实现：

```shell
url=https://github.com/pytorch/vision
commit_id=f15f4e83f06f7e969e4239c06dc17c7c9e7d731d
model_name=MobileNetV2
```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据 

  | 输入数据  | 数据类型    | 大小                             | 数据排布格式 |
  |-------|---------|--------------------------------|--------|
  | input | FLOAT16 | batchsize x 3 x height x width | NCHW   |

- 输出数据

  | 输出数据   | 大小                    | 数据类型    | 数据排布格式 |
  |--------|-----------------------|---------|--------|
  | logits | batchsize x num_class | FLOAT16 | ND     |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下依赖

  **表 1**  版本配套表

| 配套                    | 版本              | 
|-----------------------|-----------------| 
| CANN                  | 7.0.RC1.alph003 | -                                                       |
| Python                | 3.9.11          |                                                           
| PyTorch               | 2.0.1           |
| torchVison            | 0.15.2          |-
| timm                  | 0.6.13          | -                                                         |
| tqdm                  | -               | -                                                         |
| Ascend-cann-torch-aie | -               |
| Ascend-cann-aie       | -               |
| 芯片类型                  | Ascend310P3     | -                                                         |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 安装CANN包

 ```shell
 chmod +x Ascend-cann-toolkit_6.3.RC2.alpha002_linux-aarch64.run 
 ./Ascend-cann-toolkit_6.3.RC2.alpha002_linux-aarch64.run --install
 ```

下载Ascend-cann-torch-aie和Ascend-cann-aie得到run包和压缩包

## 安装Ascend-cann-aie

  ```shell
  chmod +x Ascend-cann-aie_6.3.T200_linux-aarch64.run
  ./Ascend-cann-aie_6.3.T200_linux-aarch64.run --install
  cd Ascend-cann-aie
  source set_env.sh
  ```

## 安装Ascend-cann-torch-aie

  ```shell
  tar -zxvf Ascend-cann-torch-aie-6.3.T200-linux_aarch64.tar.gz
  pip3 install torch-aie-6.3.T200-linux_aarch64.whl
  ```

## 安装其他依赖

  ```shell
  pip3 install -r requirements.txt
  ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型使用ImageNet 50000张图片的验证集，请前往[ImageNet官网](https://image-net.org/download.php)下载数据集
    ```
    ├── ImageNet
    |   ├── val
    |   |    ├── ILSVRC2012_val_00000001.JPEG
    │   |    
    │   |    ├── ......
    |   ├── val_label.txt
    ```

2. 数据预处理，步骤如下：

  ```shell
  # 参考https://github.com/pytorch/examples/tree/main/imagnet/extract_ILSVRC.sh的处理。
  mkdir -p imagenet/val && mv ILSVRC2012_img_val.tar imagenet/val/ && cd imagenet/val && tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
  bash valprep.sh
  ```

执行`valprep.sh`之后，相同类别的图片会被放到同一个目录，当前路径下会生成`./imagenet/val`数据集目录。

## 模型推理<a name="section741711594517"></a>

1. 获取模型权重文件。
   
   [https://download.pytorch.org/models/mobilenet_v2-b0353104.pth](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth)

2. 导出原始torchscript模型，用于编译优化。
    ```shell
    python3 export.py --input_path=./mobilenet_v2-b0353104.pth --output_path=./mobilenet_v2.ts --batch_size=4 --image_size=224
    ```
    
    参考命令中各选项均为代码默认值，用户可以按需修改，但要注意`batch_size`和`image_size`要与后面推理相对应

    导出模型后，会在当前目录下生成`mobilenet_v2.ts`文件。

3. 运行模型评估脚本，测试ImageNet验证集推理精度
    ```shell
    python3 eval.py --device_id=0 --model_path=./mobilenet_v2.ts --data_path=./imagenet/val --batch_size=4 --image_size=224
    ```
    运行结束后，可以看到命令行打印如下信息，说明 top1 和 top5 精度分别为 71.8700% 和 90.3220%。
    ```
    top1 is 71.87, top5 is 90.322, step is 12500
    ```

## 性能测试

可以参考以下命令进行性能测试

```shell
python3 perf.py --device_id=0 --model_path=./mobilenet_v2.ts --batch_size=4 --image_size=224
```

运行结束后可以看到命令行打印如下信息，说明性能为5675.0914 fps。

```shell
Model loaded successfully.
Start compiling model.
Model compiled successfully.
warmup done
Start performance test.
FPS: 5675.0914
```

# 模型推理性能及精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用torch-aie推理计算，精度参考下列数据。

| 芯片型号  | Batch Size | 数据集      | 精度                                  | 性能             |
|-------|------------|----------|-------------------------------------|----------------|
| 310P3 | 4          | ImageNet | top-1: 71.8700% <br>top-5: 90.3220% | FPS: 5675.0914 |

