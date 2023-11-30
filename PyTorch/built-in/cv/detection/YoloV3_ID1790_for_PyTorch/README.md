# YoloV3 for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

YOLOv3借鉴了YOLOv1和YOLOv2，在保持YOLO家族速度的优势的同时，提升了检测精度，尤其对于小物体的检测能力。YOLOv3算法使用一个单独神经网络作用在图像上，将图像划分多个区域并且预测边界框和每个区域的概率。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo
  commit_id=3e902c3afc62693a71d672edab9b22e35f7d4776
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/detection
  ```


# 准备训练环境

## 准备环境

 - 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

   **表 1**  版本支持表

   | Torch_Version      | 三方库依赖版本                  |
   | :--------: | :----------------------------------------------------------: |
   | PyTorch 1.5 | torchvision==0.2.2.post3；pillow==8.4.0 |
   | PyTorch 1.8 | torchvision==0.9.1；pillow==9.1.0 |
   | PyTorch 1.11 | torchvision==0.12.0；pillow==9.1.0 |
   | PyTorch 2.1 | torchvision==0.16.0；pillow==9.1.0 |

 - 环境准备指导。

   请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

 - 安装依赖。

   在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
   ```
   pip install -r 1.5_requirements.txt  # PyTorch1.5版本
   pip install -r 1.8_requirements.txt  # PyTorch1.8版本
   pip install -r 1.11_requirements.txt  # PyTorch1.11版本
   pip install -r 2.1_requirements.txt  # PyTorch2.1版本
   ```
   > **说明：** 
   >只需执行一条对应的PyTorch版本依赖安装命令。

 - 安装 `mmcv` 与 `mmdet`。
   1. 进入解压后的源码包根目录。

      ```
      cd /${模型文件夹名称} 
      ```

   2. 编译 `MMCV`。
   
      ```
      cd ../
      git clone -b v1.2.7 --depth=1 https://github.com/open-mmlab/mmcv.git

      export MMCV_WITH_OPS=1
      export MAX_JOBS=8
      source ${模型文件夹名称}/test/env_npu.sh

      cd mmcv
      python3 setup.py build_ext
      python3 setup.py develop
      pip3.7 list | grep mmcv
      ```
   
      将 `mmcv_need` 目录下的文件替换到 `mmcv` 的安装目录下。
      安装完后执行以下命令：
      
      ```
      cd ${模型文件夹名称} 
      cp -f mmcv_need/_functions.py ../mmcv/mmcv/parallel/
      cp -f mmcv_need/builder.py ../mmcv/mmcv/runner/optimizer/
      cp -f mmcv_need/data_parallel.py ../mmcv/mmcv/parallel/
      cp -f mmcv_need/dist_utils.py ../mmcv/mmcv/runner/
      cp -f mmcv_need/distributed.py ../mmcv/mmcv/parallel/
      cp -f mmcv_need/optimizer.py ../mmcv/mmcv/runner/hooks/
      ```
      
      或者运行 `env_set.sh` 脚本，进行 `MMCV` 的安装

      ```
      bash env_set.sh
      ```
   
   3. 安装 `mmdet`。

      执行以下命令，安装 `mmdet`。
      ```
      cd YoloV3_for_PyTorch
      pip3.7 install -r requirements/build.txt
      pip3.7 install -v -e .
      pip3.7 list | grep mm
      ```

   4. 编译安装 `Opencv-python`。

      为了获得最好的图像处理性能，***请编译安装 `opencv-python` 而非直接安装***。编译安装步骤如下：

      ```
      export GIT_SSL_NO_VERIFY=true
      git clone https://github.com/opencv/opencv.git
      cd opencv
      mkdir -p build
      cd build
      cmake -D BUILD_opencv_python3=yes -D BUILD_opencv_python2=no -D PYTHON3_EXECUTABLE=/usr/local/python3.7.5/bin/python3.7m -D PYTHON3_INCLUDE_DIR=/usr/local/python3.7.5/include/python3.7m -D PYTHON3_LIBRARY=/usr/local/python3.7.5/lib/libpython3.7m.so -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/python3.7.5/lib/python3.7/site-packages/numpy/core/include -D PYTHON3_PACKAGES_PATH=/usr/local/python3.7.5/lib/python3.7/site-packages -D PYTHON3_DEFAULT_EXECUTABLE=/usr/local/python3.7.5/bin/python3.7m ..
      make -j$nproc
      make 
      ```

## 准备数据集

1. 获取数据集。

   用户自行获取 `coco2017` 数据集，上传至服务器任意目录下并解压，数据集目录结构参考如下所示。

   ```shell script
   ├── coco2017 #根目录
         ├──train2017 #训练集图片，约118287张
         ├──val2017 #验证集图片，约5000张
         │──annotations #标注目录             
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=real_data_path     # 单卡精度
     bash ./test/train_performance_1p.sh --data_path=real_data_path    # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=real_data_path     # 8卡精度
     bash ./test/train_performance_8p.sh --data_path=real_data_path    # 8卡性能   
     ```

   - 多机多卡性能数据获取流程。

     ```shell
     1. 安装环境
     2. 开始训练，每个机器请按下面提示进行配置
       bash ./test/train_performance_multinodes.sh --data_path=数据集路径 --batch_size=单卡batch_size --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
     ```
   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --optimizer.lr                      //初始学习率
   --data.samples_per_gpu              //每个设备上的训练批次大小
   --npu_ids                           //训练设备卡号
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME     | Acc@1 |  FPS | Epochs | AMP_Type | Torch_Version |
|:--------:| :---: | :---: |:-----:| :------: | :-------:    |
| 1p-竞品V | - | - | 273 | - | 1.5 |
| 8p-竞品V | - | - | 273 | - | 1.5 |
| 1p-NPU   | -     |  91.21  | 273      |       O2 |       1.8   |
| 8p-NPU   | 0.255 | 914.29  | 273    |       O2 |       1.8   |


# 版本说明

## 变更
2023.2.3：更新基线。

2022.9.28：更新内容，重新发布。

2022.3.18：首次发布。

## FAQ

1. hipcc检查问题。

    若在训练模型时，有报"which: no hipcc in (/usr/local/sbin:..." 的日志打印问题，而hipcc是amd和nvidia平台需要的，npu并不需要。
    
    建议在torch/utils/cpp_extension.py文件中修改代码，当检查hipcc时，抑制输出。

    将 hipcc = subprocess.check_output(['which', 'hipcc']).decode().rstrip('\r\n') 修改为 hipcc = subprocess.check_output(['which', 'hipcc'], stderr=subporcess.DEVNULL).decode().rstrip('\r\n')

2. invalid pointer 问题。

    在Ubuntu、x86服务器上训练模型，有时会报invalid pointer的错误。

    解决方法：去掉scikit-image这个依赖，pip3 uninstall scikit-image

3. 报 No module named 'mmcv._ext' 问题。

   在宿主机上训练模型，有时会报No module named 'mmcv._ext'问题，或者别的带有mmcv的报错。

   解决方法：这一般是因为宿主机上安装了多个版本的mmcv，而训练脚本调用到了不匹配yolov3模型使用的mmcv，因此报mmcv的错误。

   为了解决这个问题，建议在启动训练脚本前，先导入已经安装的符合 `yolov3` 模型需要的 `mmcv` 路径的环境变量。`export PYTHONPATH=mmcv的路径:$PYTHONPATH` 。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md









