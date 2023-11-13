# ResNeXt-50模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)





# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

ResNeXt50是一种用于图像分类的卷积神经网络，这个模型的默认输入尺寸是224×224，有三个通道。通过利用多路分支的特征提取方法，提出了一种新的基于ResNet残差模块的网络组成模块，并且引入了一个新的维度cardinality。该网络模型可以在于对应的ResNet相同的复杂度下，提升模型的精度（相对于最新的ResNet和Inception-ResNet)）同时，还通过实验证明，可以在不增加复杂度的同时，通过增加维度cardinality来提升模型精度，比更深或者更宽的ResNet网络更加高效。


- 参考实现：

  ```
  url=https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
  commit_id=78ed10cc51067f1a6bac9352831ef37a3f842784
  model_name=ResNeXt
  ```
  




## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 1000 | ND           |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

| 配套                    | 版本              | 
|-----------------------|-----------------| 
| CANN                  | 6.3.RC2.alph002 |                                                       |
| Python                | 3.9.11          |                                                           
| PyTorch               | 2.0.1           |
| torchVison            | 0.15.2          |                                                      |
| Ascend-cann-torch-aie | -               |
| Ascend-cann-aie       | -               |
| 芯片类型                  | Ascend310P3     |                      



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

## 获取源码<a name="section4622531142816"></a>
1. 安装依赖。

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

1. 获取权重文件。

   ```
      wget https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
   ```

2. 导出torchscript模型文件，用于使用AIE进行模型编译和优化。
       
    运行ResNeXt50_pth2ts.py脚本。
    ```
        python ResNeXt50_pth2ts.py ./resnext50_32x4d-7cdf4587.pth ./resnext50.ts
    ```

    会在当前目录下获得resnext50.ts文件。

3. 运行模型性能评估脚本，此处使用随机数据评估模型。
    
    评估使用AIE推理引擎进行模型推理的性能分为静态shape和动态shape，对应的脚本为calculate_cost_static.py与calculate_cost_dynamic.py。

    评估静态输入，运行calculate_cost_static.py脚本，会打印出吞吐量。

    ```
        python calculate_cost_static.py --ts_model=./resnext50.ts
    ```

    动态输入，运行calculate_cost_dynamic.py脚本，会打印出吞吐量。

    ```
        python calculate_cost_dynamic.py --ts_model=./resnext50.ts
    ```
   
4. 运行模型精度评估脚本，acc_eval.py，测试ImageNet验证集推理精度。

    ```
        python acc_eval.py --model_path=./resnext50.ts --data_path=./imagenet/val
    ```
   
5. 运行结束后，可以看到命令行打印如下信息，说明 top1 和 top5 精度分别为 71.35% 和 90.502%。
    ```
    top1 is 71.35, top5 is 90.502, step is 50000
    ```
   
6. 如果需要更高精度，可以尝试修改精度评估脚本acc_eval.py中的数据前处理部分，例如修改数据normalize的标准差std的值来进行调整。

7. 因编译模型时默认使用的‘optimization_level=0’参数设置，即不使用AOE优化，若相同BatchSize时达不到下表中的性能，需要先将
calculate_cost_static文件中的 ‘optimization_level’ 设置为1，再运行性能评估脚本，然后再将其设置为2，再运行性能评估脚本。
# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用torch-AIE推理计算，静态shape性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度(top1) | 精度（top5） | 性能(吞吐量) |
| --------- |------------| ---------- |----------|----------|---------|
|     Ascend310P3      | 1          |     imagenet       | 71.35%   | 90.5%    | 820     |
|     Ascend310P3      | 4          |     imagenet       | 71.35%   | 90.5%    | 1468    |
|     Ascend310P3      | 8          |     imagenet       | 71.35%   | 90.5%    | 1588    |
|     Ascend310P3      | 16         |     imagenet       | 71.35%   | 90.5%    | 1556    |
|     Ascend310P3      | 32         |     imagenet       | 71.35%   | 90.5%    | 1607    |
|     Ascend310P3      | 64         |     imagenet       | 71.35%   | 90.5%    | 1309    |
