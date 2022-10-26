# WDL-for-PyTorch

## 1. 参考论文
[DLRS 2016][Wide & Deep Learning for Recommender Systems](http://www.arxiv.org/pdf/1606.07792.pdf)

## 2. 准备数据集
### 2.1 下载[criteo数据集](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)，解压
### 2.2 数据预处理
  - 将criteo_preprocess.py拷贝到数据集路径
  - 进入数据集目录并执行 `python3 criteo_preprocess.py train.txt`
  - 执行上述脚本后，在当前路径下生成 `wdl_trainval.pkl, wdl_test.pkl, wdl_infer.txt`

## 3. 安装依赖
- NPU环境首先安装run包，以及NPU版本的torch与apex
- `pip3.7 install -r requirements.txt`

## 4. 导入环境变量
- `source ./test/env.sh`

## 5. 配置数据集路径
- 进入test目录，路径下脚本包括：
```
env.sh                       # NPU环境变量
docker_start.sh              # 训练容器启动脚本
docker_start_infer.sh        # 推理容器启动脚本
train_full_1p.sh             # 单卡训练精度脚本
train_full_8p.sh             # 8卡训练精度脚本
train_performance_1p.sh      # 单卡训练性能脚本
train_performance_8p.sh      # 8卡训练性能脚本
```
- 修改对应训练脚本中的 `data_path` 为数据集目录路径如 `data_path=/data/criteo/`
  
## 6. 执行训练脚本
- 单卡训练执行，如 `bash ./test/train_full_1p.sh --data_path=/data/criteo/ --device_id=npu卡id`
- 8卡训练执行，如 `bash ./test/train_full_8p.sh --data_path=/data/criteo/`