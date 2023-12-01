## MatrixVT for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述

针对目前BEV中更有优势的Lift-Splat类方法中关键模块（Vision Transformation），MatrixVT实现了非常优雅的优化，在保持模型性能（甚至略微提高）的同时，能大幅降低计算量和内存消耗。

- 参考实现：

  ```
  url=https://github.com/Megvii-BaseDetection/BEVDepth/
  commit_id=d78c7b58b10b9ada940462ba83ab24d99cae5833
  ```
- 适配昇腾 AI 处理器的实现：

    ```
    url=https://gitee.com/ascend/ModelZoo-PyTorch.git
    code_path=PyTorch/built-in/autonoumous_driving/MatrixVT/
    ```
# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 2.1 | - |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

  - 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  ```shell
  pip install -r requirements.txt
  ```

- 安装mmcv==1.x（如果环境中有mmcv，请先卸载再执行以下步骤）。
  ```shell
  git clone -b 1.x https://github.com/open-mmlab/mmcv
  cd mmcv
  
  安装完成后将源码根目录下面code_for_change中的setup.py替换到mmcv中的setup.py
  
  MMCV_WITH_OPS=1 pip install -e . -v
  
  安装完成mmcv之后，进入{mmcv_install_path}/mmcv/ops/deform_conv.py文件夹，修改deform_conv.py文件，修改内容如下：
  导入torch_npu
  修改第56行，将torch.npu_deformable_conv2dbk修改为torch_npu.npu_deformable_conv2dbk
  
  修改modulated_deform_conv.py，修改内容如下：
  导入torch_npu
  修改第59行，将torch.npu_deformable_conv2d修改为torch_npu.npu_deformable_conv2d
    
  cd ../
  ```

- 安装mmdet3d==1.0.0rc4（如果环境中有mmdet，请先卸载再执行以下步骤）。
  ```shell
  pip install mmdet3d==1.0.0rc4
  ```
  

- 返回源码目录
  ```shell
  python setup.py develop
  ```
- 源码替换
  ```shell
  pip show pytorch_lightning
  ```
  找到pytorch_lightning的安装路径，然后将源码根目录下面code_for_change中的文件替换掉pytorch_lightning安装路径下对应的文件。
  ```shell
  cp ./code_for_change/subprocess_script.py {pytorch_lightning_install_path}/strategies/launchers/subprocess_script.py
  cp ./code_for_change/training_epoch_loop.py {pytorch_lightning_install_path}/loops/epoch/training_epoch_loop.py
  cp ./code_for_change/types.py {pytorch_lightning_install_path}/utilities/types.py
  ```

### 准备数据集
**Step 0.** 请用户自行到nuScenes官网下载数据集。

**Step 1.** 将数据集的路径软链接到 `./data/`。
```
ln -s [nuscenes root] ./data/
```
数据集结构如下所示。
```
MatrixVT for PyTorch
├── data
│   ├── nuScenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```
**Step 2.** 在源码根目录下进行数据集预处理。
```
python scripts/gen_info.py
python scripts/gen_depth_gt.py

```

### 支持单机8卡训练
**Step 1.** 进入源码根目录。

```
cd /${模型文件夹名称}
```

**Step 2.**  运行训练脚本。

```
当前配置下，不需要修改train_full_8p.sh中的ckpt路径，如果涉及到epoch的变化，请用户根据路径自行配置ckpt。

bash ./test/train_full_8p.sh # 8卡精度
bash ./test/train_performance_8p.sh # 8卡性能

```

### Benchmark
**精度**

| Exp    |  mAP   |  mATE  |  mASE  |  mAOE  |  mAVE  |  mAAE  |  NDS   |
|--------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| GPU_8p | 0.3322 | 0.6495 | 0.2767 | 0.4918 | 0.8780 | 0.2289 | 0.4137 | 
| NPU_8p | 0.3265 | 0.6609 | 0.2720 | 0.5212 | 0.8621 | 0.2195 | 0.4097 |

**性能**

| Exp    |   FPS   |
|--------|:-------:|
| GPU_8p | 45.6188 | 
| NPU_8p | 39.1468 | 

### FAQ
**1.** 报错scikit_learning.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block的问题，解决方案为：
  ```
  导入环境变量
  export LD_PRELOAD={libgomp-d22c30c5.so.1.0.0_path}/libgomp-d22c30c5.so.1.0.0:$LD_PRELOAD
  ```
**2.** 如果使用pip安装mmdet3d失败，请采用源码安装的方式。源码下载地址为：
  ```
  https://github.com/open-mmlab/mmdetection3d/releases/tag/v1.0.0rc4
  ```
 请下载链接中的源码压缩包，自行手动安装。同时修改mmdet3d的源码，进入mmdet3d的安装路径，修改内容有:
  ```
  1) cd /${mmdet3d-1.0.0rc4的安装路径}/mmdet3d/
     将__init__.py中的文件第41行mmcv_maximum_version = '1.7.0'修改为mmcv_maximum_version = '1.7.1'
  2) cd /${mmdet3d-1.0.0rc4的安装路径}/requirements/
     删除runtime.txt中的numba==0.53.0这一行。
  ```

