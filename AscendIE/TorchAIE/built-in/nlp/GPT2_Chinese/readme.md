# GPT2 Chinese模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******





# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

GPT-2 模型只使用了多个Masked Self-Attention和Feed Forward Neural Network，并且由多层单向Transformer的解码器构成，本质上是一个自回归模型。其中自回归的意思是指，每次产生新单词后，将新单词加到原输入句后面，作为新的输入句。而单向是指只会考虑在待预测词位置左侧的词对待预测词的影响。


- 参考实现：

  ```
  url=https://github.com/Morizeyao/GPT2-Chinese
  commit_id=bbb44651be8361faef35d2a857451d231b5ebe14
  model_name=ACL_PyTorch/built-in/nlp/GPT2_for_Pytorch
  ```

> <font size=4 color=red>说明：所有脚本都在GPT2的仓下运行</font>

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input_ids    | int64 | batchsize x 512 | ND         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output  | FLOAT16  | batchsize x 512 x 21128 | ND           |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 23.0.RC3  |
  | CANN                                                         | 7.0.RC1 | -                                                            |
  | Python                                                       | 3.9.0   | -                                                            |
  | PyTorch                                                      | 2.0.1   | -                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/Morizeyao/GPT2-Chinese       
   cd GPT2-Chinese            
   git reset --hard bbb44651be8361faef35d2a857451d231b5ebe14
   ```

2. 安装依赖。

   ```
   pip3 install -r requirement.txt
   ```

3. 获取配置文件
   
   从[这里](https://pan.baidu.com/s/16x0hfBCekWju75xPeyyRfA#list/path=%2F)下载配置文件,提取码`n3s8`，并把`pytorch_model.bin`放到`model`下面

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
> 提示：请遵循数据集提供方要求使用。
   本模型支持wiki_zh_2019验证集。用户需自行获取[数据集](https://pan.baidu.com/share/init?surl=22sax9QujO8SUdV3jH5mTQ)，提取码`xv7e`。将解压后的数据放在data下，其目录结构如下：

   ```
   data     
   └── wiki_zh
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。
      
   ```
   python3 pre_data.py
   ```
   结果保存在`data/tokenized_eval`

## 模型推理<a name="section741711594517"></a>

1. 模型编译。

   使用torch aie将模型权重文件pytorch_model.bin转换为.pt文件。
   ```
   python3 bin2pth.py --batch_size=1
   ```
   如果环境为第一次运行，可尝试使用aoe进行调优，参考如下：
   ```
   python3 bin2pth.py --batch_size=1 --optimization_level=1
   python3 bin2pth.py --batch_size=1 --optimization_level=2
   ```   

2. 开始推理验证。
   ```
   python3 compare_loss.py --batch_size=1
   ```
   
# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度指标（Loss）| 性能 |
| :------: | :--------: | :----: | :--: | :--: |
|     310P3     |      1      |   wiki_zh_2019     |  16.5   |  149    |
|     310P3     |      4      |   wiki_zh_2019     |   16.5   |  188    |
|     310P3     |      8      |   wiki_zh_2019     |   16.5   |   189   |
|     310P3     |      16      |   wiki_zh_2019     |   16.5   |   189   |
|     310P3     |      32      |   wiki_zh_2019     |   16.5   |   185   |
|     310P3     |      64      |   wiki_zh_2019     |   16.5   |    181  |

> 注：衡量精度的指标为验证集平均交叉熵损失（Cross-Entropy Loss），数值越低越好。