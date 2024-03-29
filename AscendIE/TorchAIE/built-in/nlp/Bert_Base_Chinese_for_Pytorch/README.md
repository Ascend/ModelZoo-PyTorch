# Bert_Base_Chinese模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集(请遵循数据集提供方要求使用)](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


## 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

`BERT`来自 Google 的论文`Pre-training of Deep Bidirectional Transformers for Language Understanding`，`BERT` 是`Bidirectional Encoder Representations from Transformers`的首字母缩写，整体是一个自编码语言模型。`Bert_Base_Chinese`是`BERT`模型在中文语料上训练得到的模型。

   [论文地址](https://arxiv.org/abs/1810.04805)
   [Bert模型解读](https://zhuanlan.zhihu.com/p/248017234)
   [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)


  ```shell
  url=https://huggingface.co/bert-base-chinese
  commit_id=38fda776740d17609554e879e3ac7b9837bdb5ee
  mode_name=Bert_Base_Chinese
  ```

### 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据       | 数据类型 | 大小                | 数据排布格式 |
  | -------------- | -------- | ------------------- | ------------ |
  | input_ids      | INT64    | batchsize x seq_len | ND           |
  | attention_mask | INT64    | batchsize x seq_len | ND           |
  | token_type_ids | INT64    | batchsize x seq_len | ND           |

   输入含义：
  - input_ids 就是编码后的词，即将句子里的每一个词对应到一个数字。
  - token_type_ids 第一个句子和特殊符号的位置是0，第二个句子的位置是1（含第二个句子末尾的 [SEP]），如果只有一个句子，那就全是0.
  - attention_mask 填充pad的位置是0，其他位置是1.
  - special_tokens_mask : if adding special tokens, this is a list of [0, 1], with 0 specifying special added tokens and 1 specifying sequence tokens. 特殊符号的位置是1,其他位置是0.

Bert 模型本身的输出是以下表格信息：
  | 输出数据 | 大小                              | 数据类型 | 数据排布格式 |
  | -------- | --------------------------------- | -------- | ------------ |
  | output   | batch_size x seq_len x vocab_size | FLOAT32  | ND           |

   输出含义：
   模型输出代表的是输入文本,大小为`batchsize x seq_len`，经过bert处理后得到的对应特征向量，每个字对应一个长度为`vocab_size`大小的特征向量，所以总的输出是`batch_size x seq_len x vocab_size`特征矩阵。

   在Bert模型的输出后面接一个`argmax`算子，求出每个字对应一个长度为`vocab_size`大小的特征向量中最大值的下标，通过该下标去词表当中查询得到对应的字,即模型预测的概率最大的字 与 真实的标签进行对比，以此计算模型的精度。
所以在trace的时候，模型定义如下：
   ```
   class RefineModel(torch.nn.Module):
      def __init__(self, tokenizer, model_path, config_path, device="cpu"):
         super(RefineModel, self).__init__()
         self._base_model = build_base_model(tokenizer, model_path, config_path, device)

      def forward(self, input_ids, attention_mask, token_type_ids):
         x = self._base_model(input_ids, attention_mask, token_type_ids)
         return x[0].argmax(dim=-1)
   ```

所以当前模型的输出信息为：
- 输出数据
  | 输出数据 | 大小                 | 数据类型 | 数据排布格式 |
  | -------- | -------------------- | -------- | ------------ |
  | output   | batch_size x seq_len | INT32    | ND           |

以下以`zhwiki-latest-pages-articles_validation.txt`文件的输入为例，介绍`bert-base-chinese`模型的输入输出

下面是`zhwiki-latest-pages-articles_validation.txt`的一行输入，该行有230个文本字符。
```
法国蜜蜂航空（French Bee），原稱法國藍色航空，是一家法国专营长航线的廉价航空公司，总部位于首都巴黎。法国蓝色航空的枢纽机场是巴黎-奥利机场，主要目的地是世界各国的度假胜地。因與捷藍航空對於藍色一詞有所爭議 ，2018更名為法國蜜蜂航空 。 ，法國蜜蜂航空正在运营或即将通航的航点如下： 法國蜜蜂航空目前与下述航司签署了联运协议： 法國蜜蜂航空与其姊妹航司加勒比海航空与法国国家铁路签署了班號共用协议。 直至2019年8月，法國蜜蜂航空擁有以下飛機： 
```

通过查询`bert-base-chinese/tokenizer.json`文件
``` json
"[CLS]":101
"法":3791
"国":1744
...
```
文本`法国蜜蜂航空`，对应的`input_ids={3791,  1744,  6057,  6044,  5661,  4958}`

特殊符号：
``` json
"[UNK]":100,"[CLS]":101,"[SEP]":102,"[MASK]":103
```

```
input_ids= tensor([[  101,  3791,  1744,  6057,  6044,  5661,  4958,   103,   100,   100,
          8021,  8024,   103,   103,   103,  1751,   103,  5682,  5661,  4958,
          8024,  3221,   671,  2157,  3791,  1744,   683,  5852,  7270,  5661,
          5296,  4638,   103,   817,  5661,  4958,   103,  1385,  8024,   103,
          6956,   855,   754,   103,   103,  2349,  7944,   511,  3791,  1744,
          5905,   103,  5661,  4958,   103,  3364,  5294,  3322,  1767,  3221,
           103,  7944,   118,  1952,  1164,  3322,  1767,  8024,   103,  6206,
           103,   103,  1765,  3221,   103,  4518,  1392,  1744,  4638,  2428,
           969,  5526,  1765,   511,  1728,  5645,   103,   103,  5661,  4958,
          2205,  3176,  5965,  5682,   671,   103,  3300,   103,   103,  6359,
          8024,  8271,  3291,   103,  4158,  3791,  1751,  6057,  6044,  5661,
          4958,   511,  8024,  3791,  1751,  6057,   103,  5661,   103,  3633,
           103,  6817,  5852,  2772,  1315,  2199,   103,  5661,  4638,   103,
          4157,  1963,   678,  8038,  3791,  1751,  8199,  6044,  5661,  4958,
          8943,   103,   680,   678,  6835,  5661,   103,  5041,  5392,   749,
          5468,  6817,  1291,   103,   103,  3791,  1751,  6057,   103,  5661,
          4958,   103,  1071,   103,  1987,  5661,  1385,  4347,  1239,  3683,
          3862,  5661,  4958,   680,  3791,  1744,  1744,  2157,  7188,  6662,
          5041, 20459,   749,  4408,  5998,  1066,   103,  1291,  6379,   511,
          4684,  5635,  9160,  2399,   129,  3299,  8024,  3791,  1751,  6057,
          6044,  5661,  4958,  3075,  3300,   809,   678,   103,  3582,  8038,
           102,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0]])
input_ids.shape= torch.Size([1, 384])
token_type_ids= tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
token_type_ids.shape= torch.Size([1, 384])
attention_mask= tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
attention_mask.shape= torch.Size([1, 384])
labels= tensor([[-100, -100, -100, -100, -100, -100, -100, 8020, -100,  100, -100, -100,
         1333, 4935, 3791, -100, 5965, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, 2442, -100, -100, -100,
         1062, -100, -100, 2600, -100, -100, -100, 7674, 6963, -100, -100, -100,
         -100, -100, -100, 5682, -100, -100, 4638, -100, -100, -100, -100, -100,
         2349, -100,  118, -100, -100, 3322, -100, -100,  712, -100, 4680, 4638,
         -100, -100,  686, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, 2949, 5965, -100, -100, -100, -100, -100, -100, -100, 6270,
         -100, 2792, 4261, -100, -100, -100, -100, 1399, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, 6044, -100, 4958, -100,
         1762, -100, -100, -100, -100, -100, 6858, -100, -100, 5661, -100, -100,
         -100, -100, -100, -100, 6057, -100, -100, -100, 4680, 1184, -100, -100,
         -100, -100, 1385, -100, -100, -100, -100, -100, -100, 6379, 8038, -100,
         -100, -100, 6044, -100, -100,  680, -100, 1992, -100, -100, -100, 1217,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, 5392, -100, -100, -100, -100, 4500, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, 7606, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]])
labels.shape= torch.Size([1, 384])
output= tensor([[[ -8.0061,  -7.9699,  -7.9604,  ...,  -6.8805,  -7.1705,  -7.0561],
         [ -7.3573,  -7.2314,  -7.2836,  ...,  -5.3851,  -4.8930,  -4.6018],
         [-11.8066,  -9.5953,  -9.6808,  ...,  -4.0184,  -3.1902,  -5.5859],
         ...,
         [ -6.8115,  -7.3968,  -7.7546,  ...,  -4.5425,  -6.6174,  -6.3858],
         [ -7.0039,  -7.2036,  -7.6012,  ...,  -4.3573,  -5.8063,  -5.7739],
         [ -6.3926,  -7.5099,  -6.8297,  ...,  -4.4578,  -7.3841,  -6.7582]]],
       grad_fn=<ViewBackward0>)
output.shape= torch.Size([1, 384, 21128])
logits= tensor([[8024, 3791, 1744, 6057, 6044, 5661, 4958, 8020,  100,  100, 8021, 8024,
         1348, 4935, 3791, 1751, 5965, 5682, 5661, 4958, 8024, 3221,  671, 2157,
         3791, 1744,  683, 5852, 7270, 5661, 5296, 4638, 2442,  817, 5661, 4958,
         1062, 1385, 8024, 2600, 6956,  855,  754, 3791, 1744, 2349, 7944,  511,
         3791, 1744, 5905, 5682, 5661, 4958, 4638, 3364, 5294, 3322, 1767, 3221,
         2349, 7944,  118, 1952, 1164, 3322, 1767, 8024,  712, 6206, 4680, 4638,
         1765, 3221,  686, 4518, 1392, 1744, 4638, 2428,  969, 5526, 1765,  511,
         1728, 5645, 6057, 1751, 5661, 4958, 2205, 3176, 5965, 5682,  671, 6270,
         3300, 2792, 4261, 6359, 8024, 3125, 3291, 1399, 4158, 3791, 1751, 6057,
         6044, 5661, 4958,  511, 8024, 3791, 1751, 6057, 6044, 5661, 4958, 3633,
         2466, 6817, 5852, 2772, 1315, 2199, 2458, 5661, 4638, 4294, 4157, 1963,
          678, 8038, 3791, 1751, 6057, 6044, 5661, 4958, 4680, 2399,  680,  678,
         6835, 5661, 4958, 5041, 5392,  749, 5468, 6817, 1291, 6379, 8038, 3791,
         1751, 6057, 6044, 5661, 4958, 1350, 1071, 1992, 1987, 5661, 1385, 1217,
         1239, 3683, 3862, 5661, 4958,  680, 3791, 1744, 1744, 2157, 7188, 6662,
         5041, 5392,  749, 4408, 5998, 1066,  775, 1291, 6379,  511, 4684, 5635,
         9160, 2399,  129, 3299, 8024, 3791, 1751, 6057, 6044, 5661, 4958, 3075,
         3300,  809,  678, 7606, 3582, 8038, 3791,  511, 8024, 8024, 8024, 8024,
         8038, 3791, 8020, 8020, 8020, 8020, 8024, 8024, 1071, 1066, 1071, 7606,
         4500, 7606,  775, 7606, 8038, 8024, 8024, 8024, 8024, 8024, 8024, 8024,
         8024, 8024, 1071, 8024, 2347, 2347, 5392, 5392, 1071, 5392, 7606,  678,
         7606, 3791,  511, 3791, 3791, 8024, 8020, 8020, 8020, 8020, 8020, 8020,
         8020, 8024, 8024, 5080, 4935, 3791, 8024, 8024, 3791, 1751, 5965, 6044,
         8020, 4958, 8020, 8020, 8020,  100,  100,  100, 8024, 1333, 8024, 5080,
         4935, 1333, 3791, 4935, 5965, 5965, 5965, 6044, 5965, 8024, 8024, 8020,
         8020, 8020, 8020,  100, 8020,  100, 8024, 8024, 8024, 1348, 4935, 3791,
         4935, 3791, 5965, 5965, 6044, 5965, 4958, 8024, 8024, 8024, 8024, 1184,
         1071, 1066, 5661,  749, 5392, 5661, 4638, 7606,  686, 1217,  686, 1217,
         1217, 1217, 1744, 4638, 2428,  969, 5526, 1765, 1469, 5392,  738, 5392,
          749, 5661, 4408, 7313, 5468,  775, 5661, 1217, 1217, 1217, 8024, 5635,
         5635, 8271, 8024,  680,  680, 1744, 6044, 6044, 6044, 4638, 1765, 1071,
         1071, 3791, 5401, 6057, 7607,  686,  686,  686, 5645, 3791, 3176, 4638]])
logits.shape= torch.Size([1, 384])

```

``` python
import numpy as np
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

tokenizer_kwargs = {
    'cache_dir': None,
    'use_fast': True,
    'revision': 'main',
    'use_auth_token': None
}
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese/', **tokenizer_kwargs)

max_seq_length = 384
max_seq_length = min(max_seq_length, tokenizer.model_max_length) # tokenizer.model_max_length= 1000000000000000019884624838656

print('tokenizer.model_max_length=', tokenizer.model_max_length)
print('max_seq_length=', max_seq_length)

text = '法国蜜蜂航空（French Bee），'
encoded = tokenizer.encode(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
        return_special_tokens_mask=True
    )

print('encode=', encoded)
print('tokenizer.decode(out)=', tokenizer.decode(encoded))
```
输出如下：
```bash
encode= [101, 3791, 1744, 6057, 6044, 5661, 4958, 8020, 100, 100, 8021, 8024, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
tokenizer.decode(out)= [CLS] 法 国 蜜 蜂 航 空 （ [UNK] [UNK] ） ， [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
```

```python
out = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
        return_special_tokens_mask=True
    )

# print('out=', out)

for k, v in out.items():
    print(k, ':', v)
```

```bash
input_ids : [101, 3791, 1744, 6057, 6044, 5661, 4958, 8020, 100, 100, 8021, 8024, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
token_type_ids : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
attention_mask : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
special_tokens_mask : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```


## 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                  | 版本             |
| --------------------- | ---------------- |
| CANN                  | 7.0.RC1.alpha003 |
| Python                | 3.9              |
| torch                 | 2.0.1+cpu        |
| torchvision           | 0.15.2+cpu       |
| Ascend-cann-torch-aie | 6.3rc2           |
| Ascend-cann-aie       | 6.3rc2           |
| 芯片类型              | Ascend310P3      |


## 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

### 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git                  # 克隆仓库的代码
   git checkout master                                                      # 切换到对应分支
   cd AscendIE/TorchAIE/built-in/nlp/Bert_Base_Chinese_for_Pytorch          # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

2. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```shell
   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/bert-base-chinese
   ```

### 准备数据集(请遵循数据集提供方要求使用)<a name="section183221994411"></a>
1. 获取原始数据集。
   [zhwik数据集官网](https://dumps.wikimedia.org/)，[License链接](https://dumps.wikimedia.org/legal.html)
   请在遵守数据集License的前提下使用。

   如果你想重新处理zhwiki的原始数据，可按照以下步骤操作。

   下载zhwiki原始数据：

    ```
    wget https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2 --no-check-certificate
    ```

    解压得到zhwiki-latest-pages-articles.xml

    ```
    pbzip2 -dk zhwiki-latest-pages-articles.xml.bz2
    ```

   下载预处理脚本：

   ```shell
    wget https://github.com/natasha/corus/raw/master/corus/third/WikiExtractor.py
   ```

   使用WikiExtractor.py提取文本，其中extracted/wiki_zh为保存路径，建议不要修改：

   ```
   python3 WikiExtractor.py zhwiki-latest-pages-articles.xml -b 100M -o extracted/wiki_zh
   ```

   将多个文档整合为一个txt文件，在本工程根目录下执行

   ```
   python3 WikicorpusTextFormatting.py --extracted_files_path extracted/wiki_zh --output_file zhwiki-latest-pages-articles.txt
   ```

   最终生成的文件名为zhwiki-latest-pages-articles.txt (也可直接采用处理好的文件)

   从中分离出验证集：

   ```shell
   python3 split_dataset.py zhwiki-latest-pages-articles.txt zhwiki-latest-pages-articles_validation.txt
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行preprocess.py脚本，完成预处理。

   ```shell
   # 输入参数：${input_patgh} ${model_dir} ${save_dir} ${seq_length}
   python3 preprocess.py ./zhwiki-latest-pages-articles_validation.txt ./bert-base-chinese ./input_data/ 384
   ```

   - 参数说明：第一个参数为zhwiki数据集分割得到验证集文件路径，第二个参数为源码路径（包含模型配置文件等），第三个参数为输出预处理数据路径，第四个参数为sequence长度。

### 模型推理<a name="section741711594517"></a>

1. 模型转换

   1. 获取权重文件

      获取权重文件：[pytorch_model.bin](https://huggingface.co/bert-base-chinese/blob/main/pytorch_model.bin)，移动到`bert-base-chinese`目录下：

      ```shell
      mv pytorch_model.bin bert-base-chinese
      ```


    2. trace模型，得到`bert_base_chinese.pt`
        ```shell
        python3 export_trace_model.py ./bert-base-chinese/ ./bert_base_chinese.pt 384
        ```


    3. 使用torch-aie接口进行模型推理，测试性能与精度
        ```
        python3 run_torch_aie.py
        ```


## 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

精度：

|       模型        |  Pth精度   | om离线推理精度 | torch-aie 推理精度 |
| :---------------: | :--------: | :------------: | :----------------: |
| Bert-Base-Chinese | Acc:77.96% |   Acc:77.96%   |    Acc: 77.90%     |

性能：

|       模型        | BatchSize | om离线推理性能(经过onnx优化)纯模型推理 | torch-aie性能(11.13号版本)端到端推理 | torch-aie性能(11.13号版本)纯模型推理 |
| :---------------: | :-------: | :------------------------------------: | :----------------------------------: | :----------------------------------: |
| Bert-Base-Chinese |     1     |          5.168ms(193.47 it/s)           |          13.96ms(71.6 it/s)           |          12.90ms（77.2 it/s）          |


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md


# 日志
1. [CANN 日志配置](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/70RC1alpha003/troublemanage/logreference/logreference_0017.html)

ASCEND_GLOBAL_LOG_LEVEL
功能描述
设置应用类日志的全局日志级别及各模块日志级别。
取值为：
0：对应DEBUG级别。
1：对应INFO级别。
2：对应WARNING级别。
3：对应ERROR级别。
4：对应NULL级别，不输出日志。
其他值为非法值。
```
export ASCEND_GLOBAL_LOG_LEVEL=1
```

ASCEND_SLOG_PRINT_TO_STDOUT
功能描述
是否开启日志打屏。开启后，日志将不会保存在log文件中，而是将产生的日志直接打屏显示。
取值为：
0：关闭日志打屏，即日志采用默认输出方式，将日志保存在log文件中。
1：开启日志打屏。
其他值为非法值。
```
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

2. AIE 日志配置
   配置文件位置：`/usr/local/Ascend/aie/latest/conf/aie.conf`
```
[log]
logFileLevel=info
logStdoutLevel=error
maxHistory=7
totalSizeCap=102400
maxFileSize=20480
```

3. Torch-AIE 日志配置
日志打屏和日志等级分别通过"TORCH_AIE_LOG_PRINT_TO_STDOUT"与"TORCH_AIE_LOG_LEVEL"环境变量设置
TORCH_AIE_LOG_PRINT_TO_STDOUT：0为不打屏；1为打屏；默认为不打屏
TORCH_AIE_LOG_LEVEL：0为debug级别；1为info级别；2为warn级别；3为error级别；默认为error级别
使用示例：
```commandline
export TORCH_AIE_LOG_PRINT_TO_STDOUT=1 && export TORCH_AIE_LOG_LEVEL=0 #打屏，日志级别为debug
```
日志文件位于默认路径：~/ascend/log/torch_aie_log/