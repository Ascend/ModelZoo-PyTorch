# CPM-Finetune

本仓库为CPM模型的 fine-tune 代码仓库，可以用于模型 fine-tune 的训练/测试。目前只支持了 ChID 中文成语填空数据集的实现。[[项目首页](https://cpm.baai.ac.cn)] [[模型下载](https://cpm.baai.ac.cn/download.html)] [[技术报告](https://arxiv.org/abs/2012.00413)]

同时，该仓库也提供了 ChID 数据集 zero-shot setting 下测试代码。

ChID 数据集来源于论文 [ChID: A Large-scale Chinese IDiom Dataset for Cloze Test](https://github.com/TsinghuaAI/CPM-1-Finetune). 本仓库中使用 Json 格式.

由于bpe_3w_new文件夹较大已删除，如需使用请到如下地址下载相应文件
- https://github.com/TsinghuaAI/CPM-1-Generate/tree/main/bpe_3w_new

## 1 环境准备

**bios环境设置**：

| 选项         | 设置        | 备注                                                   |
| ------------ | ----------- | ------------------------------------------------------ |
| NUMA         | Enable      | 开启numa配置：非统一内存访问架构                       |
| Power Policy | Performance | 菜单路径：Advanced > Performance config > Power Policy |


**安装基础依赖：**

- 请参照requirements安装相关依赖包。


- 确认pt_set_env.sh中的环境变量设置：

  检查路径是否按照安装方式设置（主包安装/分包安装）；

​       检查aicpu的路径ASCEND_AICPU_PATH，需要指定到aicpu包的安装路径下。

​       默认安装的情况下，如果是整包安装，需要设置为

```
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/${cpu_type}
```

​       如果是分包，需要

```
export ASCEND_AICPU_PATH=/usr/local/Ascend/
```

## 2 Zero-Shot

### 2.1 数据预处理

```[bash]
python3 preprocess_chid_zeroshot.py --data_dir ${PATH_TO_DATA_DIR} --tokenizer_path ${PATH_TO_TOKENIZER} --output_dir ${PATH_TO_OUTPUT}
```

该文件会将每个候选的成语填入文章相应的空白中，每个空白生成10个新的候选文章。最终，该文件生成的数据格式为：

```[python]

{
    "contents": [
        [8, 15, ....],
        ....
    ], # 所有样本经过 bpe 分词之后 token 对应的 id。
    "sids": [
        0,
        0,
        ...
        1,
        1,
        ...
    ], # 每个生成出的候选文章对应原来样本的编号
    "cids": [
        0,
        1,
        2,
        ...
        9,
        0,
        1,
        ...
    ], # 每个生成出的候选文章对应的成语的编号
    "labels": [
        3,
        2,
        ...
    ], # 每个原样本的正确答案编号（0~9之间的整数）
}
```

与处理完成后，指定的输出目录下会生成 test.json 文件。

### 2.2 Zero-Shot 测试

1、单机单卡自动化测试。

在test/eval_1p.sh文件里更改数据路径和模型路径如下：

```
DATA_DIR=""
CHECKPOINT_PATH=""
```

然后运行以下语句开始推理：

```
bash eval_1p.sh
```

2、单机单卡和单机8卡手动测试。

```[bash]
脚本zero-shot_chid_large_1p.sh用于单机单卡，zero-shot_chid_large_8p.sh用于单机8卡。
```

运行脚本之前，需要先将脚本中以下变量更改为实际的路径：

```[bash]
DATA_DIR # 预处理后数据的目录
CHECKPOINT_PATH # 预训练结束后模型的路径
RESULTS_DIR # 训练结果的存放处
MODEL_NAME # 给模型起的名字
TOKENIZER_PATH # tokenizer 的路径
```

## 4 参考性能

|      | 单机单卡 | 单机8卡 |
| ---- | -------- | ------- |
| 精度 | 0.679    | 0.679   |
| 性能 | 4h50min  | 1h15min |

## 5 引用

```[latex]
@article{cpm-v1,
  title={CPM: A Large-scale Generative Chinese Pre-trained Language Model},
  author={Zhang, Zhengyan and Han, Xu, and Zhou, Hao, and Ke, Pei, and Gu, Yuxian and Ye, Deming and Qin, Yujia and Su, Yusheng and Ji, Haozhe and Guan, Jian and Qi, Fanchao and Wang, Xiaozhi and Zheng, Yanan and Zeng, Guoyang and Cao, Huanqi and Chen, Shengqi and Li, Daixuan and Sun, Zhenbo and Liu, Zhiyuan and Huang, Minlie and Han, Wentao and Tang, Jie and Li, Juanzi and Sun, Maosong},
  year={2020}
}
```
# 公网地址说明

代码涉及公网地址参考 public_address_statement.md