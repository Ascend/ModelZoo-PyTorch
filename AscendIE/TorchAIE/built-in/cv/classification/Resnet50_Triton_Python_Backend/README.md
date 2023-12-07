# Resnet50

- [概述](#overview)

- [环境准备](#environment)

- [快速上手](#serviceStart)

  ******




# 概述<a name="overview"></a>

Resnet是残差网络(Residual Network)的缩写,该系列网络广泛用于目标分类等领域以及作为计算机视觉任务主干经典神经网络的一部分，典型的网络有resnet50, resnet101等。Resnet网络的证明网络能够向更深（包含更多隐藏层）的方向发展。


- 参考实现：

  ```
  url=https://github.com/pytorch/examples/tree/main/imagenet
  ```

## 输入输出数据

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output  | 1 x 1000 | FLOAT32  | ND           |



# 推理环境准备<a name="environment"></a>

- 该模型需要以下依赖

  **表 1**  版本配套表

| 配套                    | 版本              | 
|-----------------------|-----------------| 
| CANN                  | 6.3.RC2.alph002 | -                                                       |
| Python                | 3.9.0           |                                                           
| PyTorch               | 2.0.1           |
| torchVison            | 0.15.2          |-
| Ascend-cann-torch-aie | -               
| Ascend-cann-aie       | -               
| 芯片类型                  | Ascend310P3     | -                                                         |

## 拉取Triton官方镜像，启动容器
启动tritonserver容器，测试使用的版本为r22.12，根据需要添加需要配置和挂载的环境、修改容器名和映射目录等配置。
参考命令：
```
docker run -dit --name my_container \
--privileged = true \
-v /home/:/home/ \
--shm-size = 1g \
--net=host \ 
nvcr.io/nvidia/tritonserver:22.12-py3 \
/bin/bash
```

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

## 安装Python3.9
TritonServer容器内Python默认版本为3.8.10,Python backend默认版本为Python3.10, pt插件需要Python版本为3.9.x。因此需要在容器内安装python3.9版本，再重新编译使用Python3.9的Python Backend。
```
apt-get install python3.9-dev
``` 
## 安装Ascend-cann-torch-aie
 ```
 tar -zxvf Ascend-cann-torch-aie-6.3.T200-linux_aarch64.tar.gz
 pip3 install torch-aie-6.3.T200-linux_aarch64.whl
 ```

## 安装其他依赖
```
pip3 install pytorch==2.0.1
pip3 install torchVision==0.15.2
```

## 重新编译python backend
Python backend默认版本为Python3.10，重新编译更改其使用的Python版本为3.9。
### 安装依赖，拉取python backend代码
```
apt-get install rapidjson-dev libarchive-dev zlib1g-dev
git clone https://github.com/triton-inference-server/python_backend -b r22.12
```
### 适配昇腾NPU支持（可选）
如果不添加该部分适配，会默认在device 0上初始化单个模型实例。
```
git apply < npu_changes.diff
```
应用/npu/npu_changes.diff更改，添加对昇腾NPU的支持（比如设置NPU device id、配置NPU上的多模型实例device id)
### 编译Python Backend
```
cd python_backend
mkdir build && cd build
cmake -DTRITON_ENABLE_GPU=ON -DTRITON_BACKEND_REPO_TAG=r22.12 -DTRITON_COMMON_REPO_TAG=r22.12 -DTRITON_CORE_REPO_TAG=r22.12 -DCMAKE_INSTALL_PREFIX:PATH=/opt/tritonserver ..
make install
```

# 快速上手<a name="serviceStart"></a>
## Triton Python Backend推理服务
对需要支持的每个模型都需要创建model.py，说明处理config和request并返回resposne的流程。

### 创建模型库
Triton官方要求的模型库目录结构为:
```
├── resnet50
│   ├── config.pbtxt
│   └── 1
│       └── model.py
```
直接使用models/目录作为Triton的模型仓。

### 启动TritonServer

1. 获取权重文件。

前往[Pytorch官方文档](https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet50)下载对应权重，参考下载权重如下：
   
[权重](https://download.pytorch.org/models/resnet50-0676ba61.pth)


2. 修改model.py中权重文件路径.
  
3.  启动TritonServer，加载resnet50模型
    ```
    tritonserver --model-repository=./models/
    ```

### 模型推理
1. 重新用tritonserver镜像启动一个容器，或者新建窗口进入原来的容器。
2. 安装TritonClient
```
python3 -m pip install tritonclient[all]
```
3. 运行client.py，向Triton发送请求，得到推理结果。


