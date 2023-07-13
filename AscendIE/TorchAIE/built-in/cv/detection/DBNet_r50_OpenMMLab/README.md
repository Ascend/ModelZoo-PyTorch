# DBNet_r50_OpenMMLab

- [概述](#ABSTRACT)
- [环境准备](#ENV_PREPARE)
- [准备数据集](#DATASET_PREPARE)
- [快速上手](#QUICK_START)
- [模型推理性能&精度](#INFER_PERFORM)
  
***

## 概述 <a name="ABSTRACT"></a>
本模块展现的是针对openmmlab中开发的DBNet_r50模型进行了适配的昇腾pytorch插件的样例。本样例展现了如何使用mmdeploy将DBNet进行转换并通过昇腾pytorch插件将其赋予昇腾推理引擎的能力并在npu上高性能地运行。
- 模型链接
    ```
    url=https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/dbnet/README.md
    ```
- 模型对应配置文件
    ```
    url=https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py
    ```
### 输入输出数据
- 输入数据

  | 输入数据 | 数据类型 | 大小                  | 数据排布 |
  | ------- | -------- | -------------------- | ------- |
  | input   |          | bs x 3 x 1024 x 1728 | NCHW    |

- 输出数据
  
  | 输出数据 | 数据类型 | 大小              | 数据排布 |
  | ------- | -------- | ---------------- | ------- |
  | output  | Float32  | bs x 1024 x 1728 | ND      |

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
根据OpenMMLab仓库mmdeploy中的get_started.md配置mmdeploy仓库。
进入配置的mmdeploy地址并在当前路径下下载对应mmocr和mmcv（cpu版）代码仓。

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
在本地的mmdeploy地址下载本代码仓中dbnet_compile_run.py和env.sh脚本和模型权重文件。

## 准备数据集 <a name="DATASET_PREPARE"></a>>
- 数据集下载地址
  ```
    url=https://rrc.cvc.uab.es/?ch=4&com=downloads
  ```
  这里我们使用的 ICDAR2015 的500张图片的测试数据集。从链接中下载 Test Set Images 数据集并根据下方排布对数据集进行处理。
- 数据集格式
  ```
  ├── test_icdar2015_images
   |   ├── ch4_test_images
   |   |    ├── img_1.JPEG
   │   |    
   │   |    ├── ......
   |   ├── data_list.txt （必要）
  ```
  数据集路径内排布必须有 data_list.txt 文件（可以自己生成）。data_list.txt 文件中每一行是一个图片的相对路径，比如 “ch4_test_images/img_1.JPEG”。

## 快速上手 <a name="QUICK_START"></a>
一下所有命令均在mmdeploy路径下运行。
- 脚本命令
  | 命令                  | 必要 | 数据类型 | 默认值          | 描述 | 
  |-----------------------|------|---------|----------------|------| 
  | --dataset_root        | T    | str     |                | 含有data_list.txt文件的数据集路径 |
  | --trace_compile       | F    | bool    | False          | 是否需要trace并compile AIE模型 |
  | --aie_model_name      | F    | str     | "aie_model.pt" | 要保存或加载的AIE模型文件 |
  | --aie_model_save_path | F    | str     | "./"           | 要保存或加载的AIE模型文件路径 |
  | --batch_size          | F    | int     | 1              | batch size |

- trace并compile 并运行 AIE模型
  ```
    source env.sh
    python dbnet_compile_run.py --dataset_root 数据集路径 --trace_compile
  ```

- load 并运行 AIE模型
  ```
    source env.sh
    python dbnet_compile_run.py --dataset_root 数据集路径
  ```

## 模型推理性能&精度 <a name="INFER_PERFORM"></a>
| 芯片型号 | Batch Size | 数据集    | 性能 | 精度 |
|---------|------------|-----------|------|------|
| 310P3   | 1          | ICDAR2015 | 5.88qps | 与pytorch直接推理结果一致 |
| 310P3   | 8          | ICDAR2015 | 5qps | 与pytorch直接推理结果一致 |