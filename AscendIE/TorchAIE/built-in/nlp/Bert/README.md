# Bert-base

- [概述](#ABSTRACT)
- [环境准备](#ENV_PREPARE)
- [准备数据集](#DATASET_PREPARE)
- [快速上手](#QUICK_START)
- [模型推理性能&精度](#INFER_PERFORM)
  
***

## 概述 <a name="ABSTRACT"></a>
本项目为Bert-base模型在昇腾pytorch插件运行的样例，本样例展现了如何对Bert模型进行trace&导出Torchscript模型，以及在310P下利用昇腾pytorch插件运行Bert对MRPC数据集进行测试。
- 模型链接
    ```
    url=https://huggingface.co/bert-base-uncased/tree/main
    ```
- 模型对应配置文件
    ```
    url=https://huggingface.co/bert-base-uncased/blob/main/pytorch_model.bin
    ```
### 输入输出数据
- 输入数据

  | 输入数据           | 数据类型  | 大小                   | 数据排布|
  |----------------|-------|------------------------------|--------|
  | input_ids      | INT32 | batch_size x sequence_length |   ND   |
  | attention_mask | INT32 | batch_size x sequence_length |   ND   |
  | token_type_ids | INT32 | batch_size x sequence_length |   ND   |

- 输出数据
  
  | 输出数据              | 数据类型 | 大小                       | 数据排布 |
  |-------------------| - |-------------------------------------|---------|
  | last_hidden_state |  FLOAT32 | batch_size x sequence_length x 768  |    ND    |
  | pooler_output     |  FLOAT32 | batch_size x 768                    |    ND    |

## 环境准备 <a name="ENV_PREPARE"></a>
| 配套                    | 版本              | 
|-----------------------|-----------------| 
| CANN                  | 6.3.RC2.alph002 |
| Python                | 3.9             |                                                           
| Torch+cpu             | 2.0.1           |
| torchVison            | 0.15.2          |
| conda                 | 23.5.2          |
| Ascend-cann-torch-aie | --              |
| Ascend-cann-aie       | --              |
| 芯片类型                  | Ascend310P3     |
### 配置Bert运行环境
#### 环境配置步骤
首先需要配置Bert运行环境，安装环境前需要自行下载安装conda并配好权限和环境。
  ```
  conda create -n bert python=3.9 pytorch=2.0.1 torchvision -c pytorch -y
  conda activate bert
  pip3.9 install transformers
  pip3.9 install datasets
  ```
接着按照Bert安装说明进行安装，这里建议自行在[模型链接](https://huggingface.co/bert-base-uncased/tree/main)预先下载对应文件如下:
  ```
   ├── bert-base-uncased
   |   ├── config.json
   |   ├── pytorch_model.bin
   |   ├── tokenizer.json
   |   ├── tokenizer_config.json
   |   ├── vocab.txt
  ```


#### 参考链接
  ```
  url=https://huggingface.co/bert-base-uncased/blob/main/README.md
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

#### 安装torch_aie

```
tar -zxvf Ascend-cann-torch-aie-6.3.T200-linux_aarch64.tar.gz
pip3 install torch-aie-6.3.T200-linux_aarch64.whl
```

### 准备数据集
通过执行下面的脚本将mrpc数据集下载到当前路径
```
from datasets import load_dataset
dataset = load_dataset("glue", "mrpc")
dataset.save_to_disk('./')
```

### 准备脚本与必要文件
- 关键路径
  ```
   ├── work_dir
   |   ├── bert-base-uncased
   |   ├── mrpc 
   |   ├── bert_aie_eval.py
   |   ├── bert_export_ts.py
   |   ├── dynamic_bert_aie_eval.py
  ```

## 快速上手 <a name="QUICK_START"></a>

- trace 并导出Bert-base模型
  ```
    python bert_export_ts.py
  ```

- load 并运行 AIE模型（静态512）
  ```
    python bert_aie_eval.py
  ```

- load 并运行 AIE模型（动态）
  ```
    python dynamic_bert_aie_eval.py
  ```

## 模型推理性能&精度 <a name="INFER_PERFORM"></a>
| 芯片型号  | Batch Size | 数据集  | 性能          | 精度               |
|-------|------------|------|-------------|------------------|
| 310P3 | 1          | MRPC | 131 items/s | 与pytorch直接推理结果一致 |


