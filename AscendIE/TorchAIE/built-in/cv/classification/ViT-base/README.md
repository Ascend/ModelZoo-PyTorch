# ViT base

- [概述](#ZH-CN_TOPIC_0000001172161501)

- [环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

- [模型推理精度](#ZH-CN_TOPIC_0000001172201573)

  ******



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

`Transformer` 架构已广泛应用于自然语言处理领域。本模型的作者发现，Vision Transformer（ViT）模型在计算机视觉领域中对CNN的依赖不是必需的，直接将其应用于图像块序列来进行图像分类时，也能得到和目前卷积网络相媲美的准确率。

- 参考实现：

```
url=https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
mode_name = [
   vit_base_patch8_224, 
   vit_base_patch16_224, 
   vit_base_patch16_384, 
   vit_base_patch32_224,
   vit_base_patch32_384,
]
```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

   1. 对于 vit_base_patch8_224、vit_base_patch16_224 和 vit_base_patch32_224

      | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
      | -------- | -------- | ------------------------- | ------------ |
      | input    | FLOAT32  | batchsize x 3 x 224 x 224 | NCHW         |

   2. 对于 vit_base_patch16_384 和 vit_base_patch32_384

      | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
      | -------- | -------- | ------------------------- | ------------ |
      | input    | FLOAT32  | batchsize x 3 x 384 x 384 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小                  | 数据排布格式 |
  | -------- | -------- | --------              | ------------ |
  | output   | FLOAT32  | batchsize x num_class | ND           |



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
   获取模型权重。
   下载链接可参考：https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py

   模型变体较多，可按需下载。根据下表通过搜索文件名找到对应的权重文件下载地址，下载到当前目录下。

   |            模型变体|                                                                                                文件名|
   |--------------------|------------------------------------------------------------------------------------------------------|
   | vit_base_patch8_224|  B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz|
   |vit_base_patch16_224| B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz|
   |vit_base_patch16_384| B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz|
   |vit_base_patch32_224|B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz|
   |vit_base_patch32_384|  B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz|

   然后将权重文件重命名为```模型变体名称.npz```
   ```bash
   # 以 vit_base_patch8_224 为例
   mv B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz vit_base_patch8_224.npz
   ```
2. 导出原始torchscript模型，用于编译优化。
    ```
    python3 export.py --model_name vit_base_patch8_224 --checkpoint_path ./vit_base_patch8_224.npz --image_size 224
    ```
    导出模型后，会在当前目录下生成vit_base_patch8_224.ts文件。
3. 运行C++推理样例
    ```cpp
    sh build.sh
    ./build/sample ./vit_base_patch8_224.ts 224 1
    ```
    在上面的命令中，"./vit_base_patch8_224.ts"是原始torchscript模型路径，"224"是模型输入的图片尺寸大小，"1"是batch size。  
    运行结束后，可以看到命令行打印“[SUCCESS] AIE inference result is the same as JIT!"， 
    说明AIE推理结果与torchscript原始模型推理结果一致。若不一致，可先在数据集上测试精度，只要精度达标，可忽略模型输出的差异。
4. 运行模型评估脚本，测试ImageNet验证集推理精度
    ```
    python3 eval.py --model_path ./vit_base_patch8_224.ts --data_path ./imagenet/val --batch_size 1 --image_size 224
    ```
    运行结束后，可以看到命令行打印如下信息，说明 top1 和 top5 精度分别为 85.632% 和 97.764%。
    ```
    top1 is 85.632, top5 is 97.764, step is 50000
    ```
5. 测试模型推理性能
   ```shell
   # 分别将模型优化等级设置为 1 和 2，进行图优化和算子优化，达到最优性能
   python perf_test.py --model_path ./vit_base_patch8_224.ts --batch_size 1 --image_size 224 --optim_level 1
   python perf_test.py --model_path ./vit_base_patch8_224.ts --batch_size 1 --image_size 224 --optim_level 2
   ```
    运行结束后，可以看到命令行打印如下信息，说明推理性能约为 59 fps。
   ```
   FPS: 58.79
   ```

# 模型推理性能及精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用torch-aie推理计算，精度性能参考下列数据。

| 芯片型号 | 模型变体 | Batch Size | 数据集 | 精度                                   | 性能（fps） |
| --------- |-----|------------| ---------- |------------------------------------- |---------|
| 310P3 | vit_base_patch8_224    | 1          | ImageNet | top-1: 85.632% <br>top-5: 97.764% | 59      |

