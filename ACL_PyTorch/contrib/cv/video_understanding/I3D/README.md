# I3D模型推理指导

## 1、概述

### 1.1、模型概述

​		I3D是一种新的基于2D ConvNet 膨胀的双流膨胀3D ConvNet (I3D)。一个I3D网络在RGB输入上训练，另一个在流输入上训练，这些输入携带优化的、平滑的流信息。 模型分别训练了这两个网络，并在测试时将它们的预测进行平均后输出。深度图像分类ConvNets的过滤器和池化内核从2D被扩展为3D，从而可以从视频中学习效果良好的时空特征提取器并改善ImageNet的架构设计，甚至是它们的参数。[论文链接](https://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html)

### 1.2、代码链接

​		[代码链接](https://github.com/open-mmlab/mmaction2)

## 2、环境说明

无明确要求，以下环境仅供参考：

```shell
onnx==1.9.0
torch==1.5.0
torchvision==0.6.0
numpy==1.21.0
opencv-python==4.5.3.56
mmcv==1.3.9
```

## 3、获取数据集

首先clone仓库：

```shell
git clone https://github.com/open-mmlab/mmaction2.git
git checkout dbf5d59fa592818325285b786a0eab8031d9bc80
cd mmaction2
```

将acl_net.py放置在mmaction2目录下，将i3d_inference.py放置在tools目录下。

然后在仓库中创建目录。

```shell
mkdir -p data/kinetics400
```

获取标注文件：运行仓库中的tools/data/kinetics/download_backup_annotations.sh.将会在data/kinetics400目录下创建annotations目录。

```shell
cd tools/data/kinetics
bash download_backup_annotations.sh kinetics400
cd ../../..
```

最后获取kinetics400验证集。

|   数据集    | 验证集视频 |
| :---------: | :--------: |
| kinetics400 |   19796    |

我们使用的 Kinetics400 验证集包含 19796 个视频，用户可以从 [验证集视频](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB) 下载这些视频。
下载并解压好后，将数据集重命名为videos_val并放置于data/kinetics400目录下。

数据集准备完成。

## 4、数据预处理

若有另外需求，请参考[数据处理](docs_zh_CN/data_preparation.md)。

运行预处理脚本。将对videos_val中的所有视频进行抽帧处理，并将结果放置在data/kinetics400/rawframes_val目录下。本脚本采用Opencv对mp4格式的视频，采用4线程抽取256*256大小的RGB帧，输出格式为jpg。

```shell
bash build_rawframes.sh
```

将generate_labels.py文件放在data/kientics400目录下。运行该脚本获取验证所需要的验证文件。将生成kinetics400_label.txt和kinetics400_val_list_rawframes.txt。kinetics400_val_list_rawframes.txt即为验证时需要的文件。

```shell
python generate_labels.py
```

数据预处理完成。

## 5、模型转换

首先配置环境变量：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

本节采用的模型输入为:1x10x3x32x256x256.（`$batch $clip $channel $time $height $width` ）。实验证明，若想提高模型精度，可增加`$clip`的值，但性能会相应降低。若想使用其他维度大小的输入，请修改i3d_pth2onnx.sh和i3d_onnx2om.sh文件。由于本模型较大，batch_size只能设置为1，若大于1则会因为 Ascend 310 内存不足而报错。

首先在mmaction2目录下创建新目录checkpoints

```shell
mkdir checkpoints
```

然后下载预训练模型：[i3d_nl_not_product_r50](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/i3d)，在README里选择i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb配置，下载权重文件，并将该权重文件重命名为i3d_nl_dot_product_r50.pth，并保存在checkpoints目录下。

将pth文件转换为onnx文件：

310:
```
python3.8 tools/pytorch2onnx.py configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py checkpoints/i3d_r50_32x2x1_100e_kinetics400_rgb_20200614-c25ef9a4.pth --shape 1 30 3 32 256 256 --verify --show --output i3d.onnx --opset-version 11
```

310P:
```
python3.8 tools/pytorch2onnx.py configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py checkpoints/i3d_r50_32x2x1_100e_kinetics400_rgb_20200614-c25ef9a4.pth --shape 1 30 3 32 256 256 --verify --show --output i3d.onnx --opset-version 11
```
将onnx文件转换为om文件：


310:
```shell
bash i3d_atc.sh
```
310P:
```shell
bash i3d_atc.sh
```
得到i3d_bs1.om。模型转换完成。

## 6、离线推理

执行脚本：

```shell
bash i3d_infer.sh
```

即可获取top1_acc，top5_acc和mean_acc。



## 7、精度统计
|        | TOP1 | TOP5 | 
| :----: | :---: | :----:|
| 310精度  | 71.18% |   90.21%   |
| 310P精度 |71.19% |   90.21%   |

## 8、性能对比

Ascend 310：需要om文件。执行脚本。

```shell
xx/benchmark.x86_64 -device_id=0 -om_path=./i3d_nl_dot_bs1.om -round=30 -batch_size=1
```

需要先确定benchmark工具所在的绝对路径将上述命令中的xx替换。


Ascend 310P: 需要om文件。执行脚本。

```shell
xx/benchmark.x86_64 -device_id=0 -om_path=./i3d_bs1.om -round=30 -batch_size=1
```
需要先确定benchmark工具所在的绝对路径将上述命令中的xx替换。

GPU：只需要onnx文件。执行脚本。

```shell
trtexec --onnx=i3d_nl_dot.onnx --fp16 --shapes=0:1x30x3x32x256x256 --threads
```

|  |  310  | 310P | 310P_aoe | t4 |310P_aoe/310|310P_aoe/t4|
| :------: | :---: | :----: | :------: | :----: |:----: |:----: |
|    bs1     | 3.03 |   4.45    |    6.15    | 3.38 |2.03|1.82|
|    top1     | 0.7118 |       |    0.7119    |  |||
|  top5   | 0.9021 |      |    0.9021    |  |||


最优batch：310P大于310的1.2；310P大于t4的1.6倍，性能达标。