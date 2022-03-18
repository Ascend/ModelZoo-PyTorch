# RotatE

在数据集MuCo、MPII和MuPoTS上实现对3DMPPE_ROOTNET的训练。
- 实现参考：
```
url=https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding.git
branch=master
commit_id=2e440e0f9c687314d5ff67ead68ce985dc446e3a
```

## 3DMPPE_ROOTNET 细节

3DMPPE_ROOTNET是一个经典姿态估计网络，它可以对一张图片上多人的三维姿态进行估计，这也其名称的而来。该网络基于一种相机距离感知的自上而下的方法，从单个RGB图像进行三维多人姿态估计，它包括人物检测模块、人物绝对三维定位模块和基于三维的单人姿态估计模块。该模型取得了与最先进的三维单人姿态估计模型相当的结果。

## 环境准备

- 安装 PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- 训练数据集是FB15k-237，数据集较小。
- https://ascend-pytorch-one-datasets.obs.cn-north-4.myhuaweicloud.com/train/zip/FB15k-237.zip
- 解压数据集在`data`文件夹中，遵循以下的目录结构：
```
${RotatE}
|-- data
|   |-- FB15k-237
|   |   |-- entities.dict
|   |   |-- relations.dict
|   |   |-- test.txt
|   |   |-- train.txt
|   |   |-- valid.txt
```
- 请在`models`文件夹中遵循以下目录结构：
```
${RotatE}
|-- models
|   |-- save_path  ## 模型保存路径
|   |   |-- checkpoint_0      ## 保存的模型文件
|   |   |-- config_0.json     ## 配置的参数
|   |   |-- output_0.prof     ## 训练的prof文件
|   |   |-- train_0.log       ## 训练的日志保存在这里
|   |   |-- train_time_0.log  ## 训练每个step的时间日志

```

## 训练模型

- 运行 `apex_run.py` 进行模型训练：

```
# 1p train perf
bash test/train_performance_1p.sh

# 8p train perf
bash test/train_performance_8p.sh

# 1p train full
bash test/train_full_1p.sh

# 8p train full
bash test/train_full_8p.sh

```

## 训练结果

| MRR    | FPS       | Npu_nums | Steps   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
|    0.3354     |2874.12      | 1        | 100000      | O1       |
|       0.3252  | 18930.59       | 8        | 100000      | O1       |

# 其它说明 # 

- 运行 `demo.py`：
```
python codes/demo.py -save XX
```

