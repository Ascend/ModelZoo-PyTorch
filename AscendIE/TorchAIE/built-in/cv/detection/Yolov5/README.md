
# YOLOV5模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

   - [输入输出数据](#ZH-CN_TOPIC_0000001126281702)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理精度](#ZH-CN_TOPIC_0000001172201573)

  ******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

YOLO是一个经典的目标检测网络，将目标检测作为回归问题求解。本文旨在提供基于推理引擎的Yolov5参考样例，使用了coco2017数据集，并测试了昇腾310P芯片上的推理精度供参考，模型性能仍在持续优化中。


- 参考实现：

  ```shell
  url=https://github.com/ultralytics/yolov5/tree/v6.1
  branch=master
  commit_id=3752807c0b8af03d42de478fbcbf338ec4546a6c
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | images   | RGB_FP32 | batchsize x 3 x 640 x 640 | NCHW         |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下依赖

  **表 1**  版本配套表

| 配套                    | 版本              | 
|-----------------------|-----------------| 
| CANN                  | 6.3.RC2.alph002 | -                                                       |
| Python                | 3.9         |                                                           
| PyTorch               | 2.0.1           |
| torchVison            | 0.15.2          |-
| Ascend-cann-torch-aie | 6.3.T200           
| Ascend-cann-aie       | 6.3.T200        
| 芯片类型                  | Ascend310P3     | -                                                         |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>
## 安装CANN包

 ```
 chmod +x Ascend-cann-toolkit_6.3.RC2.alpha002_linux-aarch64.run 
./Ascend-cann-toolkit_6.3.RC2.alpha002_linux-aarch64.run --install
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

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

    ```
    git clone https://github.com/ultralytics/yolov5.git
    cd yolov5
    git checkout v6.1  # 切换到v6.1版本
    ```

2. 安装依赖。

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

本模型需要coco2017数据集，labels下载[地址](https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip)，验证集下载[地址](https://images.cocodataset.org/zips/val2017.zip)


   数据集结构如下
   ```
    coco
     |-- LICENSE
     |-- README.txt
     |-- annotations
     |   `-- instances_val2017.json
     |-- images
     |   |-- train2017
     |   |-- val2017
     |-- labels
     |   |-- train2017
     |   `-- val2017
     |-- test-dev2017.txt
     |-- train2017.txt
     |-- val2017.cache
     `-- val2017.txt
   ```

将数据集放在yolov5源码上一级路径的datasets下
```
cd yolov5
cd ..
mkdir datasets
mv coco datasets
```


## 模型推理<a name="section741711594517"></a>
1. 获取权重文件。
```
# YOLOv5s
wget https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt
# YOLOv5m
wget https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5m.pt
```
2. 生成trace模型
    ```
    cd yolov5
    # 生成yolov5s.torchscript
    python export.py
    # 生成yolov5m.torchscript
    python export.py --weights yolov5m.pt
    ```

3. 保存编译优化模型

    ```
    python export_torch_aie.py --torch-script-path ${PWD}/yolov5s.torchscript --batch-size 1
    python export_torch_aie.py --torch-script-path ${PWD}/yolov5m.torchscript --batch-size 1
     ```


2. 执行推理脚本
   ```
   cd yolov5
   vim val.py +31
   # 添加如下两行代码即可实现推理引擎模型推理
   import torch_aie
   torch_aie.set_device(0)  # 0表示所使用的NPU_ID

   # 执行推理
   batch_size=1  # 设置batch_size
   python val.py --weights yolov5sb${batch_size}_torch_aie.torchscript --data coco.yaml --img 640 --conf 0.001 --iou 0.65 --batch-size ${batch_size}
   ```
   
3.  运行C++样例
    ```
    # 请根据实际运行环境修改run.sh和build.sh中torch和torchAIE的路径
    bash build.sh
    bash run.sh
    ```
    执行结束后，会在当前路径生成C++编译优化后模型yolov5[s\m]b${batch_size}_torch_aie.torchscript。如果输出"compare pass!"，则表示和CPU上的推理结果一致。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

yolov5s模型精度如表2

| 芯片型号 | Batch Size   | 数据集 | mAP<sup>val<br>50 | mAP<sup>val<br>50-95 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|    Ascend310P3       |       1    |   coco2017         |     56.1%       |   37.1%              |
|

yolov5m模型精度如表3

| 芯片型号 | Batch Size   | 数据集 | mAP<sup>val<br>50 | mAP<sup>val<br>50-95 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|    Ascend310P3       |       1    |   coco2017         |     63.2%      |   44.8%             |
|