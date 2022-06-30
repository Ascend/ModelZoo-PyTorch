# ResNet18 Onnx模型端到端推理指导

## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[ResNet18论文](https://arxiv.org/pdf/1512.03385.pdf)  

### 1.2 代码地址
[ResNet18代码](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)  
branch:master
commit_id:7d955df73fe0e9b47f7d6c77c699324b256fc41f

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.0.3

torch == 1.5.1
torchvision == 0.6.1
onnx == 1.9.0
```

### 2.2 python第三方库

```
numpy == 1.19.2
Pillow == 8.2.0
opencv-python == 4.5.2
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 数据集预处理

-   **[数据集获取](#31-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 3.1 数据集获取
该模型使用ImageNet的5万张验证集进行测试，图片与标签分别存放在/root/datasets/imagenet/val与/root/datasets/imagenet/val_label.txt。

### 3.2 数据集预处理
1.预处理脚本imagenet_torch_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 imagenet_torch_preprocess.py resnet /root/datasets/imagenet/val ./prep_dataset
```
### 3.3 生成数据集信息文件
1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 gen_dataset_info.py bin ./prep_dataset ./resnet18_prep_bin.info 224 224
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息

## 4 模型推理

-   **[pth转onnx模型](#41-pth转onnx模型)**  

-   **[onnx转om模型](#42-onnx转om模型)**  

### 4.1 pth转onnx模型

1.下载pth权重文件  
从https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py的第26行获取对应权重下载链接，使用wget命令下载对应权重

文件MD5sum：e0b1c919e74f9a193d36871d9964bf7d

2.ResNet18模型代码在torchvision里，安装torchvision，arm下需源码安装，参考torchvision官网
```
git clone https://github.com/pytorch/vision
cd vision
python3.7 setup.py install
cd ..
```
3.编写pth2onnx脚本resnet18_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

3.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 resnet18_pth2onnx.py ./resnet18-f37072fd.pth resnet18.onnx
```
### 4.2 onnx模型量化（可选）
1.AMCT工具包安装，具体参考[CANN 开发辅助工具指南 (推理)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)

2.数据预处理，用于量化因子矫正。当前模型为动态batch，建议用多batch_size的预处理文件矫正量化因子。
执行以下命令：
```
python3.7.5 calibration_bin.py prep_dataset calibration_bin 64
```

3.ONNX模型量化，具体参考[CANN 开发辅助工具指南 (推理)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)
在result目录下生成resnet18_deploy_model.onnx量化模型

4.量化模型验证，除onnx离线模型转换om模型命令有区别外，其余一致

### 4.3 onnx转om模型

1. 设置环境变量
    ```
    source env.sh
    ```
**说明**
>此脚本中环境变量只供参考，请以实际安装环境配置环境变量

2. 使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN 开发辅助工具指南 (推理)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)
   
   ${chip_name}可通过`npu-smi info`指令查看，例：310P3

    ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)
    ```
    atc --framework=5 --model=./resnet18.onnx --output=resnet18_bs1 --input_format=NCHW --input_shape="image:1,3,224,224" --log=debug --soc_version=Ascend${chip_name} --insert_op_conf=aipp.config --enable_small_channel=1 # Ascend310P3


    ## Int8量化（可选）
    atc --framework=5 --model=./result/resnet18_deploy_model.onnx --output=resnet18_bs1_int8 --input_format=NCHW --input_shape="image:1,3,224,224" --log=debug --soc_version=Ascend${chip_name} --insert_op_conf=aipp.config --enable_small_channel=1 # Ascend310P3
    ```

### 4.4 模型离线推理

1.设置环境变量
```
source env.sh
```
**说明**
>此脚本中环境变量只供参考，请以实际安装环境配置环境变量

2.增加benchmark.{arch}可执行权限
```
chmod u+x benchmark.x86_64
```

3.执行离线推理
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=resnet18_bs1.om -input_text_path=./resnet18_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
执行./benchmark.x86_64工具请选择与运行环境架构相同的命令。详情参考[CANN 推理benchmark工具用户指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)

4.精度验证
调用imagenet_acc_eval.py脚本与数据集标签val_label.txt比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python3.7 imagenet_acc_eval.py result/dumpOutput_device0/ /home/HwHiAiUser/dataset/imagenet/val_label.txt ./ result.json
```
第一个参数为生成推理结果所在路径，第二个参数为标签数据，第三个参数为生成结果文件路径，第四个参数为生成结果文件名称