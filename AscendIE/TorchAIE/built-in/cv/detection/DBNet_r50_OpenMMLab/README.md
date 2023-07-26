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
    url=https://github.com/open-mmlab/mmocr
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
| CANN                  | 6.3.RC2.alph002 |
| Python                | 3.9        |                                                           
| Torch+cpu             | 2.0.1           |
| torchVison            | 0.15.2          |
| conda                 | 23.5.2          |
| Ascend-cann-torch-aie | --              |
| Ascend-cann-aie       | --              |
| 芯片类型               | Ascend310P3     |
### 配置OpenMMLab运行环境
#### 环境配置步骤
首先需要配置mmocr运行环境。安装环境前需要自行下载安装conda并配好权限和环境。接着按照mmocr安装说明进行安装，这里安装时需要自行按照提示补齐缺少的库：
  ```
  conda create -n open-mmlab python=3.9 pytorch=2.0.1 torchvision -c pytorch -y
  conda activate open-mmlab
  pip3 install openmim
  git clone https://github.com/open-mmlab/mmocr.git
  cd mmocr
  mim install -e .
  pip install mmdeploy==1.2.0
  ```
接着我们需要配置mmdeploy才能将模型迁移到torchscript上。这里我们在mmocr同级路径下下载mmdeploy仓。而后我们需要为DBNet中无法直接trace的自定义算子（mmdeploy::modulated_deform_conv）构建torchscript算子：
  ```
  git clone https://github.com/open-mmlab/mmdeploy.git
  cd mmdeploy
  mkdir -p build && cd build
  cmake -DCMAKE_CXX_COMPILER=g++-9 -DMMDEPLOY_TARGET_BACKENDS=torchscript -DTorch_DIR=${Torch_DIR} ..
  make -j$(nproc) && make install
  ```
#### 参考链接
  ```
  url=https://github.com/open-mmlab/mmocr/blob/main/README_zh-CN.md
  url=https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/01-how-to-build/linux-x86_64.md
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
在本地的mmdeploy地址下载本代码仓中dbnet_compile_run.py脚本和模型权重文件。
- 关键路径
  ```
   ├── work_dir
   |   ├── mmdeploy
   |   |    ├── dbnet_sample.py
   │   |    ├── {MODEL_WEIGHT_FILE}.pt
   │   |    ├── ......
   |   ├── mmocr
  ```

## 准备数据集 <a name="DATASET_PREPARE"></a>
clone了mmocr仓后执行下面命令
```
  cd mmocr
  python tools/dataset_coverters/prepare_dataset.py icdar2015 --task textdet
```
数据集会生成在 ```mmocr/data/icdar2015``` 中。在后续运行时视情况对 ```mmocr/configs/textdet/_base_/datasets/icdar2015.py``` 中的 ```icdar2015_textdet_data_root``` 的路径做更改来保证运行脚本可以获取到数据集。

## 快速上手 <a name="QUICK_START"></a>
一下所有命令均在mmdeploy路径下运行。
- 脚本命令
  | 命令                  | 必要 | 数据类型 | 默认值          | 描述 | 
  |-----------------------|------|---------|----------------|------|
  | --trace_compile       | F    | bool    | False          | 是否需要trace并compile AIE模型 |
  | --aie_model_name      | F    | str     | "aie_model.pt" | 要保存或加载的AIE模型文件 |
  | --aie_model_save_path | F    | str     | "./"           | 要保存或加载的AIE模型文件路径 |
  | --batch_size          | F    | int     | 1              | batch size |

- trace并compile 并运行 AIE模型
  ```
    python dbnet_compile_run.py --trace_compile
  ```

- load 并运行 AIE模型
  ```
    python dbnet_compile_run.py
  ```

## 模型推理性能&精度 <a name="INFER_PERFORM"></a>
| 芯片型号 | Batch Size | 数据集    | 性能 | 精度 |
|---------|------------|-----------|------|------|
| 310P3   | 1          | ICDAR2015 | 5.88qps | 与pytorch直接推理结果一致 |
| 310P3   | 4          | ICDAR2015 | 5.25qps | 与pytorch直接推理结果一致 |
| 310P3   | 8          | ICDAR2015 | 5qps    | 与pytorch直接推理结果一致 |