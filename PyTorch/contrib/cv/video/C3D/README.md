#  C3D

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

C3D模型使用经过大规模视频数据集预训练的3D ConvNets来学习视频的时空特征，可以同时对外观和运动信息进行建模，在各种视频分析任务上，证明了其采用的3D ConvNets优于2D ConvNet特征，是一个经典的视频时空特征提取backbone网络。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/c3d/README.md
  branch=master 
  commit_id=2b6f9ac69b3609b96a514501ffe30fc90545f518
  ```


- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/video
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套        | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 硬件       | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动  | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip3 install -r requirements.txt
  ```
- 安装 mmcv（在模型源码包根目录下执行以下操作）。
  ```
  export GIT_SSL_NO_VERIFY=1
  git config --global http.sslVerify false
  git clone -b v1.3.9 --depth=1 https://github.com/open-mmlab/mmcv.git
  source ./test/env_npu.sh; cd mmcv; python3.7 setup.py build_ext; python3.7 setup.py develop
  ```
- 修改mmcv。

  ```
  cp ./additional_need/mmcv/distributed.py  ./mmcv/mmcv/parallel/
  cp ./additional_need/mmcv/test.py  ./mmcv/mmcv/engine/
  cp ./additional_need/mmcv/dist_utils.py  ./mmcv/mmcv/runner/
  cp ./additional_need/mmcv/optimizer.py  ./mmcv/mmcv/runner/hooks/
  cp ./additional_need/mmcv/epoch_based_runner.py ./mmcv/mmcv/runner/
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ucf101等，将获取好的数据集上传至在源码包根目录下新建的"data/"文件夹下并解压。

   以ucf101数据集为例，数据集目录结构参考如下所示。


    ```
    data
    ├── ucf101
    │   ├── ucf101_{train,val}_split_{1,2,3}_rawframes.txt
    │   ├── ucf101_{train,val}_split_{1,2,3}_videos.txt
    │   ├── annotations
    │   ├── rawframes
    │   │   ├── ApplyEyeMakeup
    │   │   │   ├── v_ApplyEyeMakeup_g01_c01
    │   │   │   │   ├── img_00001.jpg
    │   │   │   │   ├── img_00002.jpg
    │   │   │   │   ├── ...
    │   │   │   │   ├── flow_x_00001.jpg
    │   │   │   │   ├── flow_x_00002.jpg
    │   │   │   │   ├── ...
    │   │   │   │   ├── flow_y_00001.jpg
    │   │   │   │   ├── flow_y_00002.jpg
    │   │   ├── ...
    │   │   ├── YoYo
    │   │   │   ├── v_YoYo_g01_c01
    │   │   │   ├── ...
    │   │   │   ├── v_YoYo_g25_c05
    ```

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
     bash ./test/train_full_1p.sh
     bash ./test/test_1p.sh
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh
     bash ./test/test_8p.sh
     ```


   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                         //数据集路径
   --batch_size                        //训练批次大小，默认为3
   --validate                          //启动验证
   --rank_id                           //训练卡id
   --seed                              //种子设定
   ```

   训练完成后，权重文件保存在当前路径下，并在test/output中输出模型训练精度和性能信息。
   

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 | FPS    | Epochs | AMP_Type | Torch_version |
| ------- | ----- | :----- | ------ | :------- | ------------- |
| 1p-竞品 | -     | -      | -      | -        | -             |
| 8p-竞品 | -     | -      | -      | -        | -             |
| 1p-NPU  | -     | 59.155 | 1      | O2       | 1.5           |
| 1p-NPU  | -     | 63.39  | 1      | O2       | 1.8           |
| 8p-NPU  | 80.44 | 450.783| 30     | O2       | 1.5           |
| 8p-NPU  | 81.42 | 474.6  | 30     | O2       | 1.8           |


# 版本说明

## 变更

2022.12.07：更新pytorch1.8版本，重新发布。

2021.02.14：首次发布。

## 已知问题

无。
