# VideoPose3D  模型 Pytorch 离线推理
## 1. 环境准备
1. 必要的软件依赖
    - Pytorch == 1.5.0
    - torchvision == 0.5.0
    - msame 软件，安装在当前目录下
    - numpy
2. 获取、修改与安装开源软件代码  
在当前目录下，进行以下操作  
```
git clone https://github.com/facebookresearch/VideoPose3D.git
cd VideoPose3D
git reset 1afb1ca0f1237776518469876342fc8669d3f6a9 --hard
patch -p1 < ../vp3d.patch
mkdir checkpoint
cd ..
```
3. 获取权重文件  
将提供的 `model_best.bin` 文件放在 `.\VideoPose3D\checkpoint` 目录下
4. 获取数据集  
将提供的 `data` 文件夹放在 `.\VideoPose3D` 目录下
## 2. 离线推理  
310上执行，执行时使 `npu-smi info` 查看设备状态，确保 `device` 空闲
```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh
```
### 评测结果
| 模型 | 官网 pth 精度 | 310 离线推理精度 | 基准性能 | 310 性能 |
|:----:|:----:|:----:|:----:|:----:|
|VideoPose3D conv1d bs1| 46.8 mm| 46.5 mm | 584834 fps | 409776 fps |
|VideoPose3D conv2d bs1| - | 46.6 mm | 605179 fps | 580903 fps |

备注：
- 310 离线推理使用的是我用单卡自行训练的 Model，效果好于官网
- VideoPose3D 原本代码中全程使用 conv1d（以及相应的 batchnorm1d）。考虑到转 om 后  
的 conv1d 均由 conv2d 算子实现，因此我将源码中的 conv1d 以及相应操作全部替换为 conv2d  
及其相应操作。这个修改使得在 Ascend 310 上的推理性能与原本代码在 GPU 上的推理性能持平
- 即便考虑到比较 conv2d 版本在 GPU 与 Acend310 上的性能，差距也小于二者在 conv1d  下的性能