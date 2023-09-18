# Resnet50

- [概述](#ZH-CN_TOPIC_0000001172161501)

- [环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

- [模型推理精度](#ZH-CN_TOPIC_0000001172201573)

  ******



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Swin Transformer是在Vision Transformer的基础上使用滑动窗口（shifted windows, SW）进行改造而来。
它将Vision Transformer中固定大小的采样快按照层次分成不同大小的块（Windows），每一个块之间的信息并不共通、独立运算从而大大提高了计算效率。


- 参考实现：

```
url=https://github.com/rwightman/pytorch-image-models
mode_name=SwinTransformer
```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据
* 说明：官方SwinTransformer仓的输入图片宽高相等，具体尺寸可参考配置：如swin_base_patch4_window12_384配置对应尺寸为384。   

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FLOAT16 | batchsize x 3 x height x width | NCHW         |


- 输出数据

  | 输出数据   | 大小     | 数据类型 | 数据排布格式 |
  |--------| -------- | -------- | ------------ |
  | logits | batchsize x num_class | FLOAT16 | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下依赖

  **表 1**  版本配套表

| 配套                    | 版本              | 
|-----------------------|-----------------| 
| CANN                  | 6.3.RC2.alph002 | -                                                       |
| Python                | 3.9.0           |                                                           
| PyTorch               | 2.0.1           |
| torchVison            | 0.15.2          |-
| timm                  | 0.6.13          | -                                                         |
| tqdm                  | -               | -                                                         |
| Ascend-cann-torch-aie | -               
| Ascend-cann-aie       | -
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
  ./Ascend-cann-aie_6.3.T200_linux-aarch64.run --install
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
```
# 参考https://github.com/pytorch/examples/tree/main/imagnet/extract_ILSVRC.sh的处理。
mkdir -p imagenet/val && mv ILSVRC2012_img_val.tar imagenet/val/ && cd imagenet/val && tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
bash valprep.sh
```
执行valprep.sh之后，相同类别的图片会被放到同一个目录，当前路径下会生成 ./imagenet/val 数据集目录。

## 模型推理<a name="section741711594517"></a>

1. 获取模型权重文件。
   * [swin_base_patch4_window12_384_22kto1k.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth)
   * 其他模型权重下载地址可参考: [Swin-Transformer(microsoft)](https://github.com/microsoft/Swin-Transformer)
2. 导出原始torchscript模型，用于编译优化。
    ```
    python3 export.py --model_name swin_base_patch4_window12_384 --checkpoint_path ./swin_base_patch4_window12_384_22kto1k.pth --image_size 384
    ```
    导出模型后，会在当前目录下生成swin_base_patch4_window12_384.ts文件。
3. 运行C++推理样例
    ```cpp
    sh build.sh
    ./build/sample ./swin_base_patch4_window12_384.ts 384 1
    ```
    在上面的命令中，"./swin_base_patch4_window12_384.ts"是原始torchscript模型路径，"384"是模型输入的图片尺寸大小，"1"是batch size。  
    运行结束后，可以看到命令行打印“[SUCCESS] AIE inference result is the same as JIT!"， 
    说明AIE推理结果与torchscript原始模型推理结果一致。
4. 运行模型评估脚本，测试ImageNet验证集推理精度
    ```
    python3 eval.py --model_path ./swin_base_patch4_window12_384.ts --data_path ./imagenet/val --batch_size 1 --image_size 384
    ```
    运行结束后，可以看到命令行打印如下信息，说明 top1 和 top5 精度分别为 86.4520% 和 98.0560%。
    ```
    top1 is 86.4520, top5 is 98.0560, step is 50000
    ```

# 模型推理性能及精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用torch-aie推理计算，精度参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度                                   |
| --------- |---| ---------- |------------------------------------- |
| 310P3 | 1 | ImageNet | top-1: 86.4520% <br>top-5: 98.0560% |

