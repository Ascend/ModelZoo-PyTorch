# FCOS for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

FCOS是一个全卷积的one-stage目标检测模型，相比其他目标检测模型，FCOS没有锚框和提议，进而省去了相关的复杂计算，以及相关的超参，
这些超参通常对目标检测表现十分敏感。借助唯一的后处理NMS，结合ResNeXt-64X4d-101的FCOS在单模型和单尺度测试中取得了44.7%的AP，
因其简化性在现有one-stage目标检测模型中具有显著优势。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection/tree/v2.9.0/configs/fcos
  commit_id=6c1347d7c0fa220a7be99cb19d1a9e8b6cbf7544  
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/detection
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.2.2.post3；pillow==8.4.0 |
  | PyTorch 1.8 | torchvision==0.9.1；pillow==9.1.0 |
  | PyTorch 1.11 | torchvision==0.12.0 |
  | PyTorch 2.1 | torchvision==0.16.0 |

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
  
- 准备mmcv环境。
  1. 进入源码包根目录，下载mmcv，最好是1.2.7版本的（版本要求是1.2.5以上，1.3.0以下）。
  ```
  cd FCOS
  git clone -b v1.2.7 git://github.com/open-mmlab/mmcv.git
  ```
  2. 用mmcv_need里的文件替换mmcv中对应的文件。
  ```
    cp -f mmcv_need/_functions.py ./mmcv/mmcv/parallel/
    cp -f mmcv_need/builder.py ./mmcv/mmcv/runner/optimizer/
    cp -f mmcv_need/distributed.py ./mmcv/mmcv/parallel/
    cp -f mmcv_need/data_parallel.py ./mmcv/mmcv/parallel/
    cp -f mmcv_need/dist_utils.py ./mmcv/mmcv/runner/
    cp -f mmcv_need/optimizer.py ./mmcv/mmcv/runner/hooks/
    cp -f mmcv_need/checkpoint.py ./mmcv/mmcv/runner/
    ```
  3. 以下三个文件的替换是为了在log中打印出FPS的信息，替换与否对模型训练无影响。
    ```
    cp -f mmcv_need/iter_timer.py ./mmcv/mmcv/runner/hooks/
    cp -f mmcv_need/base_runner.py ./mmcv/mmcv/runner/
    cp -f mmcv_need/epoch_based_runner.py ./mmcv/mmcv/runner/
    ```
  4. 推荐使用conda管理。
    ```
    conda create -n fcos --clone env  # 复制一个已经包含依赖包的环境 
    conda activate fcos
    ```
  5. 配置安装mmcv。
    ```
    cd mmcv
    export MMCV_WITH_OPS=1
    export MAX_JOBS=8
    python3 setup.py build_ext
    python3 setup.py develop
    pip3 list | grep mmcv  # 查看版本和路径
    ``` 
  6. 配置安装mmdet。
    ```
    cd Fcos
    pip3 install -r requirements/build.txt
    python3 setup.py develop
    pip3 list | grep mmdet  # 查看版本和路径
    ```
  7. 修改apex中的113行，主要是为了支持O1，参考路径root/archiconda3/envs/fcos/lib/python3.7/site-packages/apex/amp/utils.py。
    ```
    if cached_x.grad_fn.next_functions[1][0].variable is not x:
    ```
    改成:
    ```
    if cached_x.grad_fn.next_functions[0][0].variable is not x:
    ```

## 准备数据集

1. 请用户自行准备好数据集，包含训练集、验证集和标签三部分，可选用的数据集又COCO、PASCAL VOC数据集等。
2. 上传数据集到data文件夹，以coco2017为例，数据集在`data/coco`目录下分别存放于train2017、val2017、annotations文件夹下。
3. 当前提供的训练脚本中，是以coco2017数据集为例，在训练过程中进行数据预处理。 数据集目录结构参考如下：

   ```
   ├── coco2017
         ├──annotations
              ├── captions_train2017.json
              ├── captions_val2017.json
              ├── instances_train2017.json
              ├── instances_val2017.json
              ├── person_keypoints_train2017.json
              └── person_keypoints_val2017.json
             
         ├──train2017  
              ├── 000000000009.jpg
              ├── 000000000025.jpg
              ├── ...
         ├──val2017  
              ├── 000000000139.jpg
              ├── 000000000285.jpg
              ├── ...
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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/         # 精度训练
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 性能训练
     ```
     
   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/         # 精度训练
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 性能训练   
     ```
     

   - 多机多卡训练

     多机多卡性能数据获取流程，在每个节点上执行：

     ```
     bash ./test/train_performance_multinodes.sh --data_path=数据集路径 --batch_size=单卡batch_size --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP   
     ```

  --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   --data_path                         //数据集路径
   --device_id                         //npu卡号  
   --batch-size                        //训练批次大小
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表


|  NAME  | CPU_Type | Acc@1 |  FPS  | Epochs | AMP_Type | Loss_Scale | Torch_Version |
|:------:|:--------:|:-----:|:-----:|:------:| :------: |:----------:|:----------:|
| 1p-竞品V |   X86    | 12.6  | 19.2  |   1    | O1       |  dynamic   |      1.5   |
| 8p-竞品V |   X86    | 36.2  | 102.0 |   12   | O1       |  dynamic   |      1.5   |
| 1p-Npu |   非ARM   | 16.4  | 3.19  |   1    | O1       |    32.0    |      1.8   |
| 8p-Npu |   非ARM   | 36.2  | 44.81 |   12   | O1       |    32.0    |      1.8   |
| 8p-Npu |   ARM    | 36.2  | 35.69 |   12   |   O1       |    32.0    |      1.8   |


# 版本说明

## 变更

2022.12.21：更新Readme。


## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
