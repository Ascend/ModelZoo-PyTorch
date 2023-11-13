# Albert模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

ALBERT是BERT 的“改进版”，主要通过通过Factorized embedding parameterization和Cross-layer parameter sharing两大机制减少参数量，得到一个占用较小的模型，对实际落地有较大的意义，不过由于其主要还是减少参数，不影响推理速度。

  ```
  url=https://github.com/lonePatient/albert_pytorch
  branch=master
  commit_id=46de9ec
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据       | 数据类型 | 大小                      | 数据排布格式 |
  | --------       | -------- | ------------------------- | ------------ |
  | input_ids      | INT64    | batchsize x seq_len       | ND           |
  | attention_mask | INT64    | batchsize x seq_len       | ND           |
  | token_type_ids | INT64    | batchsize x seq_len       | ND           |

  说明：该模型默认的seq_len为128。

- 输出数据

  | 输出数据 | 大小               | 数据类型 | 数据排布格式 |
  | -------- | --------           | -------- | ------------ |
  | output   | batch_size x class | FLOAT32  | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下固件与插件

  **表 1**  版本配套表

| 配套                                                            | 版本    | 
| ------------------------------------------------------------    | ------- | 
| 固件与驱动                                                       | 23.0.RC1  | 
| CANN                                                            | 7.0.RC1.alpha003 | 
| Python                                                          | 3.9.11   | 
| PyTorch                                                         | 2.0.1 | 
| Torch_AIE                                                       | 6.3.rc2 |
| 芯片类型                                                         | Ascend310P3 |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git        # 克隆仓库的代码
   git checkout master         # 切换到对应分支
   cd AscendIE/TorchAIE/built-in/nlp/Albert              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

2. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```
   git clone https://github.com/lonePatient/albert_pytorch.git
   cd albert_pytorch
   git checkout 46de9ec
   patch -p1 < ../albert.patch
   cd ../
   ```

## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。

   本模型采用[SST-2数据集](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip)，解压到 `albert_pytorch/dataset/SST-2`文件夹下

   数据集目录结构请参考：
   ```
   ├──dataset
        ├──SST-2
            ├──original/
            ├──dev.tsv
            ├──train.tsv
            ├──test.tsv
   ```
## 准备模型<a name="section183221994411"></a>
1. 准备预训练权重文件

   下载[预训练权重文件](https://drive.google.com/open?id=1byZQmWDgyhrLpj8oXtxBG6AA52c8IHE-)，并解压到albert_pytorch/prev_trained_model/albert_base_v2下
   预训练权重目录结构请参考：
   ```
   ├──prev_trained_model
        ├──albert_base_v2
            ├──30k-clean.model
            ├──30k-clean.vocab
            ├──config.json
            ├──pytorch_model.bin

1. 准备训练好的模型

   下载[训练好的模型](https://pan.baidu.com/s/1G5QSVnr2c1eZkDBo1W-uRA )（提取码：mehp ）并解压到albert_pytorch/outputs/SST-2。
   训练好的模型目录结构请参考：
   ```
   ├──outputs
        ├──SST-2
            ├──-SST-2.log
            ├──checkpoint_eval_result.txt
            ├──config.json
            ├──pytorch_model.bin
            ├──training_args.bin

## 模型推理<a name="section741711594517"></a>

1. 编译模型。
    
    使用PyTorch将原始模型首先trace成torchscript模型，然后使用torch_aie编译成.pt文件。
    ```
    # 以bs32，seq128为例
    python3.9 export_albert_aie.py --batch_size=32 --max_seq_length=128 --compare_cpu
    ```
    
    - 执行以上命令将在会把编译好的模型存储值当前目录下的albert_seq128_bs32.pt文件，使用--compare_cpu参数则脚本还会验证编译后的模型与原始torch模型的输出是否一致。 

2. 推理验证。

    将run_aie_eval.py脚本拷贝至albert_pytorch目录下并进入该目录:

    ```
    cp run_aie_eval.py ./albert_pytorch
    cd albert_pytroch
    ```

    执行推理，验证模型的精度和吞吐量。
        
    ```
    # 以推理bs32，seq128模型为例
    python3.9 ./run_aie_eval.py --aie_model_dir=../albert_seq128_bs32.pt --model_type=SST --model_name_or_path="./prev_trained_model/albert_base_v2" --task_name="SST-2" --data_dir="./dataset/SST-2" --spm_model_file="./prev_trained_model/albert_base_v2/30k-clean.model" --output_dir="./outputs/SST-2/" --do_lower_case --max_seq_length=128 --batch_size=32
    ```

    - 推理完成后会输出模型在数据集上的分类准确率以及单位时间内推理的样本数量（吞吐率）。若要推理不同bs和seq配置的模型，只需要更改--aie_model_dir、--max_seq_length和--batch_size三个参数即可。 



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

bs32seq128对应的精度性能如下：

精度:

| 输入类型  | 芯片型号   | ACC(bs32seq128)   |
| --------- | -------- | ------------- |
| 静态      | Ascend310P3     | 92.82%         |

静态模型性能：

| 模型        | batch size   | 310P3性能   |
| :------:    | :------:  | :------:  |
| Albert base v2  | 1 | 532.61    |
| Albert base v2  | 4 | 841.79   |
| Albert base v2  | 8 | 1020.77   |
| Albert base v2 | 16 | 982.26   |
| Albert base v2 | 32 | 988.63   |
| Albert base v2 | 64 | 891.06   |

