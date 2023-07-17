# FSAF for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

FSAF为FPN每层添加anchor-free分支，包含分类与回归，在训练时，根据anchor-free分支的预测结果选择最合适的FPN层用于训练，最终的网络输出可同时综合FSAF的anchor-free分支结果以及原网络的预测结果。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection
  commit_id=2028b0c189d676ce0c7ad31f24f8a68107220855
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
  | PyTorch 1.5 | torchvision==0.2.2.post3；mmcv==1.2.7 |
  | PyTorch 1.8 | torchvision==0.9.1；mmcv==1.2.7 |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本

  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  ```
  > **说明:** 只需执行一条对应的PyTorch版本依赖安装命令。
- 安装环境。
  1. 克隆`mmcv`库并放置在当前项目目录。

     ```
     git clone -b v1.2.7 git://github.com/open-mmlab/mmcv.git
     ```
  2. 用当前项目目录下`mmcv_need`替换`mmcv`目录下的子目录`mmcv`。
     ```
     rm -rf mmcv/mmcv
     cp -r mmcv_need mmcv
     mv mmcv/mmcv_need mmcv/mmcv
     ```
  3. 配置编译mmcv。
     ```sh
     cd mmcv
     export MMCV_WITH_OPS=1
     export MAX_JOBS=8
     python3 setup.py build_ext
     python3 setup.py develop
     pip3.7.5 list | grep mmcv
     ```
  4. 安装mmdetection。
     ```sh
     cd mmdetection
     pip3.7.5 install -r requirements/build.txt
     python3 setup.py develop
     pip3.7.5 list | grep mmdet
     ```
  5. 如果遇到`apex O1`报错，尝试修改：找到代码路径`{the path of the fsaf environment in conda}/lib/python3.7/site-packages/apex/amp/utils.py`, mine is `/root/archiconda3/envs/fsaf/lib/python3.7/site-packages/apex/amp/utils.py`
     ```diff
     # change this line (line 113)
     - if cached_x.grad_fn.next_functions[1][0].variable is not x:
     # into this
     + if cached_x.grad_fn.next_functions[0][0].variable is not x:
     ```
## 准备数据集

  用户自行获取原始数据集，可选用的开源数据集包括COCO等，
   并将你的数据集放置在`$FSAF/mmdetection/data` 下。

   以COCO2017数据集为例，数据集目录结构参考如下所示。

   ```
    COCO
       ├── annotations
       │      ├── instances_val2017.json
       │      ├── instances_train2017.json
       │      ├── captions_train2017.json
       │      ├── ……
       ├── images
       │   │   ├──train2017
       |   |   │   │   ├──xxxx.jpg
       │   │   ├──val2017
       |   |   │   │   ├──xxxx.jpg
       ├── labels
       │   │   ├──train2017
       |   |   │   │   ├──xxxx.txt
       │   │   ├──val2017
       |   |   │   │   ├──xxxx.txt
       ├──test-dev2017.txt
       ├──test-dev2017.shapes
       ├──train2017.txt
       ├──……
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
     bash ./test/train_full_1p.sh  # 单卡精度

     bash ./test/train_performance_1p.sh  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh  # 8卡精度

     bash ./test/train_performance_8p.sh  # 8卡性能
     ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh real_model_path  # 8卡评测
     ```

   real_model_path参数为实际模型路径，可不传，默认为./work_dirs/fsaf_r50_fpn_1x_coco/latest.pth。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   config                         //配置文件路径
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V |   -   | 11.39 |   1   |    O1    |      1.5      |
| 8p-竞品V | 37.5  | 70.84 |  12    |   O1    |      1.5      |
|  1p-NPU  |   -   | 1.26  |   1    |    O1   |      1.5      |
|  8p-NPU  | 36.2  | 8.38  |  12    |    O1   |      1.5      |

# 版本说明

## 变更

2020.10.14：更新内容，重新发布。

2020.07.08：首次发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md