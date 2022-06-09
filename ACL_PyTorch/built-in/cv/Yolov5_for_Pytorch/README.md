# Yolov5模型推理

-   [1 环境准备](#1-环境准备)
-   [2 推理步骤](#2-推理步骤)
	-   [2.1 设置环境变量](#21-设置环境变量)
	-   [2.2 pt导出om模型](#22-pt导出om模型)
	-   [2.3 om模型推理](#23-om模型推理)
-   [3 端到端推理Demo](#3-端到端推理Demo)
-   [4 量化（可选）](#4-量化（可选）)
-   [5 FAQ](#5-FAQ)

------


## 文件说明
```
Yolov5_for_Pytorch
└── common
  ├── pth2om.sh            pth导出om模型
  ├── om_infer.py          推理导出的om模型
  └── util           pth导出om相关脚本
    ├── acl_net.py         PyACL接口
    ├── atc.sh             atc转模型脚本
    └── modify_model.py    onnx模型修改，添加NMS后处理
  └── quantize       量化相关脚本
    ├── generate_data.py   生成量化校准数据
    ├── img_info_amct.txt  用于量化校准的图片信息
    └── amct.sh            onnx模型量化
└── 2.0 / 3.1 / 4.0 / 5.0 / 6.0 / 6.1 
  ├── xx.patch             对应版本的修改patch
  └── run.sh               推理Demo
```

## 1 环境准备

### 1.1 下载pytorch源码，切换到对应分支
```shell
git clone https://github.com/ultralytics/yolov5.git
git checkout v2.0/v3.1/v4.0/v5.0/v6.0/v6.1
```

### 1.2 准备以下文件，放到pytorch源码根目录
（1）common文件夹及对应版本文件夹  
（2）对应版本的 [权重文件](https://github.com/ultralytics/yolov5/tags)  
（3）coco2017数据集val2017和label文件**instances_val2017.json**    
（4）[可选] 若需评估性能，下载benchmark工具，不同平台下载路径 [x86](https://support.huawei.com/enterprise/zh/software/255327333-ESW2000481524) / [arm](https://support.huawei.com/enterprise/zh/software/255327333-ESW2000481500)

### 1.3 安装依赖
```shell
pip install -r requirements.txt
```


## 2 推理步骤

### 2.1 设置环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2.2 pt导出om模型
运行pth2om.sh导出om模型，默认保存在output路径下，可通过`bash common/pth2om.sh -h`查看完整参数设置
```shell
bash common/pth2om.sh --version 6.1 --model yolov5s --bs 4
```

### 2.3 om模型推理
```shell
python3 common/om_infer.py --img-path=./val2017 --model=output/yolov5s_nms_bs4.om --batch-size=4
```


## 3 端到端推理Demo
对应版本文件夹下提供run.sh，可直接执行
```shell
bash v6.1/run.sh  
```


## 4 量化（可选）
安装 [量化工具](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/51RC2alpha005/developmenttools/devtool/atlasamctonnx_16_0011.html) 和 [onnx改图接口工具](https://gitee.com/peng-ao/om_gener) ，执行以下脚本导出om量化模型。
```shell
bash common/pth2om.sh --version 6.1 --model yolov5s --bs 4 --type int8 --calib_bs 16
```
说明：  
（1）量化存在精度误差，可使用实际数据集进行校准以减少精度损失。若用自己的数据集训练，要修改common/quantize/img_info_file.txt文件中的校准数据。  
（2）设置参数--type int8 --calib_bs 16可生成om量化模型，其中16为量化使用的校准数据个数，根据实际所用的校准数据修改。  


## FAQ
常见问题可参考 [FAQ](FAQ.md)

