# DBNet_r50_OpenMMLab

- [概述](#ABSTRACT)
- [环境准备](#ENV_PREPARE)
- [准备数据集](#DATASET_PREPARE)
- [快速上手](#QUICK_START)
- [模型推理性能&精度](#INFER_PERFORM)
  
***

## 概述 <a name="ABSTRACT"></a>
本模块展现的是针对openmmlab中开发的crnn模型进行了适配的昇腾pytorch插件的样例。本样例展现了如何使用mmdeploy将crnn进行转换并通过昇腾pytorch插件将其赋予昇腾推理引擎的能力并在npu上高性能地运行。
- 模型链接
    ```
    url=https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/crnn/README.md
    ```
- 模型对应配置文件
    ```
    url=https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/crnn/crnn_mini-vgg_5e_mj.py
    ```

## 环境准备 <a name="ENV_PREPARE"></a>
| 配套                   | 版本            | 
|-----------------------|-----------------| 
| CANN                  | 6.3.RC2.alph002 | 链接                                                          |
| Python                | 3.9.0           |                                                           
| PyTorch               | 2.0.0           |
| torchVison            | 0.15.1          |-
| Ascend-cann-torch-aie | --
| Ascend-cann-aie       | --
| 芯片类型               | Ascend310P3     |
### 配置OpenMMLab运行环境
运行基于openmmlab推理框架的crnn等模型进行推理前，需要提前根据openmmlab的在线指导文档安装部署mmocr与mmdeploy，分别用于文字识别与模型转换。
参考链接：
https://github.com/open-mmlab/mmocr
https://github.com/open-mmlab/mmdeploy
根据OpenMMLab仓库mmdeploy中的get_started.md配置mmdeploy仓库。
进入配置的mmdeploy地址并在当前路径下下载对应mmocr和mmdeploy代码仓。参考命令：
```
mkdir open-mmlab
cd open-mmlab
conda create -n open-mmlab python=3.9 pytorch=2.0.1
conda activate open-mmlab
pip3 install openmim
mim install mmengine
mim install "mmcv>=2.0.0rc2"
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
mim install -e .
cd ..
pip install mmdeploy==1.2.0
git clone -b main https://github.com/open-mmlab/mmdeploy.git

```

### 配置昇腾运行环境
下载对应版本的昇腾产品
#### 安装CANN包

```
chmod +x Ascend-cann-toolkit_6.3.RC2.alpha002_linux-aarch64.run 
./Ascend-cann-toolkit_6.3.RC2.alpha002_linux-aarch64.run --install
```

#### 安装推理引擎

```
chmod +x Ascend-cann-aie_6.3.T200_linux-aarch64.run
Ascend-cann-aie_6.3.T200_linux-aarch64.run --install
cd Ascend-cann-aie
source set_env.sh
```

#### 安装torch—aie

```
tar -zxvf Ascend-cann-torch-aie-6.3.T200-linux_aarch64.tar.gz
pip3 install torch-aie-6.3.T200-linux_aarch64.whl
```


### 准备脚本与必要文件
在本地的mmdeploy地址下载本代码仓中crnn_sample.py和env.sh脚本和模型权重文件。需要注意的是，脚本里的文件路径需要与实际文件路径对齐。参考目录结构：
```
open-mmlab/
|-- crnn_mini-vgg_5e_mj_20220826_224120-8afbedbb.pth
|-- crnn_sample.py
|-- mmdeploy
|   |-- CITATION.cff
|   |-- CMakeLists.txt
|   |-- ....
|   |-- third_party
|   `-- tools
`-- mmocr
    |-- CITATION.cff
    |-- ....
    |-- setup.py
    |-- tests
    `-- tools
```



## 准备数据集 <a name="DATASET_PREPARE"></a>>
clone了mmocr仓后执行下面命令
```
  cd mmocr
  python tools/dataset_coverters/prepare_dataset.py svt --task textrecog
```
数据集会生成在 ```mmocr/data/svt``` 中。
注意！通过git安装的mmocr在加载数据集时可能会出现路径错误的问题，需要对mmocr目录中以下路径的数据集配置文件进行修改(data_root改为绝对路径)
```
cd mmocr/configs/textrecog/_base_/datasets/
```

## 快速上手 <a name="QUICK_START"></a>
以下所有命令均在open-mmlab路径下运行。
- 脚本命令
  | 命令                  | 必要 | 数据类型 | 默认值          | 描述 | 
  |-----------------------|------|---------|----------------|------|
  | --trace_compile       | F    | bool    | True           | 是否需要trace并compile AIE模型 |
  | --model_path          | F    | str     | None           | 模型pth文件的路径 |
  | --batch_size          | F    | int     | 1              | batch size |
  | --img_path            | F    | str     | None              | 可以输入单张图片路径进行文字识别，不能与--dataset共用 |
  | --dataset             | F    | str     | None              | 可以输入测试集名称进行测试，不能与--img_path共用 |
  | --shape_range         | F    | bool    | false              | 用于开启动态输入，能够将不同size的图片作为模型输入 |

- 使用torch-aie编译模型并在测试集上推理，静态场景（输入宽高一致）参考命令：
  ```
    python crnn_sample.py --dataset=svt --batch_size=8
  ```
  使用torch-aie编译模型并在测试集上推理，动态场景（输入宽度可变）参考命令：
  ```
    python crnn_sample.py --dataset=cute80 --shape_range=true
  ```


## 模型推理性能&精度 <a name="INFER_PERFORM"></a>
对于mmocr的crnn模型，前处理中会将单个batch中的所有图片resize为统一宽高的输入，为对齐精度，需要开启shaperange并将batchsize设置为1，避免精度损失。
这也是openmmlab官方代码仓中采取的测试方式，可以在对应测试集上获取与官方readme一致的精度。
| 芯片型号 | Batch Size | 数据集    | 性能(吞吐量) | 精度 |
|---------|------------|-----------|------|------|
| 310P3   | 1          | cute80 | 504.5 | 0.5694(164/288) |
| 310P3   | 8          | cute80 | 1172 | - |
| 310P3   | 1          | svt | 483.9 | 0.7991(518/647) |
| 310P3   | 8          | svt | 1951 | - |
| 310P3   | 1          | svtp | 546.6 | 0.6093(393/645) |
| 310P3   | 8          | svtp | 1655 | - |