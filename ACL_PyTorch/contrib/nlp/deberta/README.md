## Deberta 模型离线推理指导

### 一、环境准备

Ascend环境: CANN 5.1.RC2

#### 1. 创建conda环境


```
conda create --name deberta python=3.7.5 -y
conda activate deberta

pip install torch==1.11.0 onnx==1.11.0 numpy==1.21.6
pip install -r requirements.txt

```

#### 2. 获取开源模型代码仓

```
git clone https://github.com/microsoft/DeBERTa.git
cd DeBERTa
git reset --hard c558ad99373dac695128c9ec45f39869aafd374e
patch -p1 < ../deberta.patch
cd ..
```


### 二、转 ONNX

1. 获取权重文件，保存到当前工作目录
    [pytorch.model-073631.bin](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/nlp/Deberta/pytorch.model-073631.bin)

    也可自己通过源码训练获得

2. 执行转 ONNX 脚本
    ```
    python3.7 pth2onnx.py --init_model ./pytorch.model-073631.bin --onnx_path ./dynamic.onnx --config ./model_config.json
    ```
    其中"init_model"表示模型加载权重的地址和名称,"onnx_path"表示转换后生成的onnx模型的存储地址和名称,[model_config.json](./model_config.json)是模型训练的时候在预训练config的基础上加上mnli任务的config拼接得到的中间文件，用于初始化模型。


### 三、转 OM

```
# 设置环境变量, 用户需使用自定义安装路径，指定为：
source /opt/npu/cann_5.1.rc205/ascend-toolkit/set_env.sh

# 执行ATC参考命令
atc --framework=5 \
--model=./dynamic.onnx \
--output=./deberta_bs${batch_size} \
--input_format=ND \
--input_shape="input_ids:${batch_size},256;input_mask:${batch_size},256" \
--soc_version=Ascend${chip_name} \
--log=error
```
说明：\\$\{batch\_size\} 表示生成不同 batch size 的 om 模型；  
\\${chip\_name}可通过 npu-smi info 指令查看，如下图标注部分。

![](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)  

### 四、数据集预处理

#### 1.获取数据集 
```
cd DeBERTa
./experiments/glue/download_data.sh ../ MNLI
cd ..
```
如果执行有问题参考源码仓./experiments/glue/download_data.sh的命令，进入源码仓分步执行下面命令
```
curl -s -J -L  https://raw.githubusercontent.com/nyu-mll/jiant/v1.3.2/scripts/download_glue_data.py -o ../glue.py
./experiments/glue/download_data.sh ../ MNLI
```
或者从此链接获取[MNLI数据集](https://ascend-pytorch-one-datasets.obs.cn-north-4.myhuaweicloud.com/infer/MNLI/MNLI.zip)
解压后数据集目录结构如下:
```
  ${data_path}
  |-- MNLI
      |-- original
      |    |-- multinli_1.0_dev_matched.jsonl
      |    |-- multinli_1.0_dev_matched.txt
      |    |-- multinli_1.0_dev_mismatched.jsonl
      |    |-- multinli_1.0_dev_mismatched.txt
      |    |-- multinli_1.0_train.jsonl
      |    |-- multinli_1.0_train.txt
      |-- dev_matched.tsv
      |-- dev_mismatched.tsv
      |-- test_matched.tsv
      |-- dev_matched.tsv
      |-- test_mismatched.tsv
      |-- train.tsv

```

#### 2. 执行数据预处理脚本
```
  python3.7 deberta_preprocess.py --datasets_path ${datasets_path}/ --pre_data_save_path ./pre_mnli_bs${batch_size} --batch_size ${batch_size}
```
参数说明：  
\\${data_path}/：数据集路径  
./pre_mnli_bs\\${batch\_size}：预处理后的 bin 文件存放路径  
\\$\{batch\_size\} 表示生成不同 batch size 的 om 模型  
> **说明：**  
> 在预处理代码里,tokenizers = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base"),需要从网上下载相关文件，可能会存在无法下载的问题。
> 解决方法：下载[pre_deberta.zip](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/nlp/Deberta/pre_deberta.zip),将里面的文件放在根目录~/.cache/huggingface/transformers下,预处理可正常运行

### 五、离线推理

1. 准备 msame 推理工具  
    msame模型推理工具，其输入是om模型以及模型所需要的输入bin文件，其输出是模型根据相应输入产生的输出文件。获取工具及使用方法可以参考[msame模型推理工具指南](https://gitee.com/ascend/tools/tree/master/msame)，将msame文件放到当前目录
2. 推理时，使用 npu-smi info 命令查看 device 是否在运行其它推理任务，提前确保 device 空闲
    ```
    # 删除之前冗余的推理文件，创建 result 文件夹
    rm -rf ./result/outputs_bs${batch_size}_om/${dataset_version}
    mkdir -p ./result/outputs_bs${batch_size}_om/${dataset_version}

    # 推理
    chmod a+x msame

    ./msame --model ./deberta_bs${batch_size}.om --input ./pre_mnli_bs${batch_size}/${dataset_version}/input_ids/,./pre_mnli_bs${batch_size}/${dataset_version}/input_mask/ --output ./result/outputs_bs${batch_size}_om/${dataset_version}
    ```
    参数说明：  
    --model：om 模型路径  
    --input：预处理后的 bin 文件存放路径  
    --output：输出文件存放路径 
    \\$\{dataset\_version\} 表示使用哪种数据集类型，取值 match 或者 mismatch  
    \\$\{batch\_size\} 表示推理使用模型的 batch size  

3. 执行数据后处理脚本
    ```
    python3.7 deberta_postprocess.py --datasets_path ${datasets_path}/ --bin_file_path ./result/outputs_bs${batch_size}_om/${dataset_version}/*/ --dataset_version ${dataset_version} --eval_save_path ./result --eval_save_file eval_bs${batch_size}_${dataset_version}.txt
    ```
    参数说明：
    --bin_file_path：推理生成的result  
    --dataset_version 表示使用哪种数据集类型，取值 match 或者 mismatch  
    --output：输出文件存放路径  
    eval_save_path 和 eval_save_file 表示输出精度数据所在的路径和文件名, 两者用于打印精度信息  
    \\$\{datasets_path\}：表示MNLI数据集，用于获得标签数据和MNLI数据集的领域类别  
    \\$\{batch\_size\} 表示推理使用模型的 batch size  
  

4. 性能测试
    使用msame工具进行纯推理，可以根据不同的batchsize和dataset_version进行纯推理。使用同一输入进行性能测试，与T4机器性能测试对比：
    ```
    ./msame --model ./deberta_bs${batch_size}.om --input ./pre_mnli_bs${batch_size}/${dataset_version}/input_ids/input_0.bin,./pre_mnli_bs${batch_size}/${dataset_version}/input_mask/input_0.bin --output ./result/perf_bs${batch_size}_om/${dataset_version} --loop 100 > msame_bs${batch_size}_${dataset_version}.txt
    ```
    说明：
    使用对应batchsize的某一输入计算100次，将结果重定向到msame_bs\\${batch_size}_\\${dataset_version}.txt文件中

**精度评测结果：**

| 模型    | 官网精度 | 310P 精度 | 基准性能 | 310P 性能 |
| ------- | ------- | -------- | -------- | -------- |
| Deberta bs1  | M=90.5/MM=90.5 | M=90.46/MM=90.71 | 2.849fps | 4.055fps |
| Deberta bs4  | M=90.5/MM=90.5 | M=90.46/MM=90.71 | 2.859fps | 4.692fps |
| Deberta bs8  | M=90.5/MM=90.5 | M=90.46/MM=90.71 | 2.821fps  | 4.639fps |
| Deberta bs16  | M=90.5/MM=90.5 | M=90.46/MM=90.71 | 2.807fps | 4.681fps |
| Deberta bs32  | M=90.5/MM=90.5 | M=90.46/MM=90.71 | 2.935fps | 4.619fps |
| Deberta bs64  | M=90.5/MM=90.5 | M=90.46/MM=90.71 | 3.065fps | 5.027fps  |