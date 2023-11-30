# Hilander

The paper proposes a hierarchical graph neural network (GNN) model that learns how to cluster a set of images into an unknown number of identities using a training set of images annotated with labels belonging to a disjoint set of identities.



## Requirements

```
1.install pytorch==1.7.0 

2.torchvision==0.8.0 

3.install tqdm

4.git clone https://github.com/yjxiong/clustering-benchmark.git 
cd clustering-benchmark
python setup.py install
第四步是install clustering-benchmark for evaluation
链接obs://cann-id2774/npu/clustering-benchmark-master

5.需要安装dgl,但是华为镜像站没有arm版本的，所以我们是使用源码编译成arm版本的再进行安装。此文件夹太大，无法上传。
链接obs://cann-id2774/npu/dgl1

6.faiss-cpu
链接obs://cann-id2774/npu/faiss_cpu-1.7.1.post3-cp37-cp37m-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

第四第五一定要python setup.py install,相关代码在try3.py

4,5,6文件放在根目录下
```

# Datasets
Download the  dataset for training and put it in the folder "data"
inat2018_train_dedup_inter_intra_1_in_6_per_class.pkl

obs://cann-id2774/dataset/

# Train

python try3.py


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md