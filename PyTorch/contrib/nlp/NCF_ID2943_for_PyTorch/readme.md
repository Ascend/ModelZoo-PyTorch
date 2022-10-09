# Neural-Collaborative-Filtering
- Paper:[Neural Collaborative Filtering.](http://dl.acm.org/citation.cfm?id=3052569) 
- Github Code:https://github.com/yihong-chen/neural-collaborative-filtering

Neural collaborative filtering(NCF), is a deep learning based framework for making recommendations. The key idea is to learn the user-item interaction using neural networks. Check the following paper for details about NCF.

## Directory structure
```
.
├── data	# 存放数据
├── data.py	#prepare train/test dataset
├── util.py	#some handy functions for model training etc.
├── metrics.py #evaluation metrics including hit ratio(HR) and NDCG
├── gmf.py #generalized matrix factorization model
├── mlp.py #multi-layer perceptron model
├── neumf.py #fusion of gmf and mlp
├── readme.md	
├── engine.py #training engine
└── train.py #entry point for train a NCF model
```
## Environment preparation
- Dataset URL:http://grouplens.org/datasets/movielens/1m/
  

## Run

python3 train.py


> 生成的模型在checkpoint文件夹中。

> 例如：gmf_factor8neg4-implict_Epoch0_HR0.0944_NDCG0.0432.model，文件名包含HR和NDCG

### Running result

| 200Epoch | HR     | NDCG   |
| -------- | ------ | ------ |
| GPU      | 0.6400 | 0.2950 |
| NPU      | 0.6407 | 0.3696 |

精度达标