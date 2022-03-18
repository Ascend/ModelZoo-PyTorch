# AdvancedEAST
AdvancedEAST-PyTorch is mainly inherited from
[AdvancedEAST](https://github.com/huoyijie/AdvancedEAST),
also we made some changes for better usage in PyTorch.
If this project is helpful to you, welcome to star.

# New features
* writen in PyTorch, easy to read and run
* change the dataset into LMDB format, reduce I/O overhead
* added precision/recall/F1_score output which is helpful when training the model
* just run `train.py` to automatically start training

# Project files
* config file: `cfg.py`, control parameters
* pre-process data: `preprocess.py` , resize image
* generate LMDB dataset: `imgs2LMDB.py`
* **[optional]** *label data: `label.py`, produce label info*
* define network: `model_VGG.py`
* define loss function: `losses.py`
* execute training: `train.py` 
* read LMDB dataset: `dataset.py`
* predict: `predict.py` and `nms.py`
* evaluate the model: `utils.py`

# Network arch
* AdvancedEast

![AdvancedEast network arch](image/AdvancedEast.network.png "AdvancedEast network arch")

[原理简介(含原理图)](https://huoyijie.cn/blog/9a37ea00-755f-11ea-98d3-6d733527e90f/play)

[后置处理(含原理图)](https://huoyijie.cn/blog/82c8e470-7562-11ea-98d3-6d733527e90f/play)

# Setup
* python 3.6.5
* PyTorch-gpu 1.4.0
* lmdb 0.98
* numpy 1.19.0
* tqdm 4.48.0
* natsort 7.0.1
* openCV 4.2.0
* shapely 1.7.0
* **[optional]** torchsummary

# Training
* tianchi ICPR dataset download
链接: https://pan.baidu.com/s/1NSyc-cHKV3IwDo6qojIrKA 密码: ye9y

* prepare training data: make data root dir(train_1000),
copy images to root dir, and copy txts to root dir, 
data format details could refer to [ICPR MTWI 2018 挑战赛二：网络图像的文本检测](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.3bcad780oQ9Ce4&raceId=231651)
* modify config params in `cfg.py`, see default values
* **[optional]** ```python preprocess.py```, resize image to 256X256, 384X384, 512X512, 640X640, 736X736, 
and train one by one could speed up training process(依次训练可以加速模型收敛)
* **[optional]** ```python imgs2LMDB.py```, generate LMDB sataset
* ```python train.py```, train entrance
* ```python predict.py -p demo/001.png```, to predict
* pretrain model download(use for further training or test)
链接: 链接: https://pan.baidu.com/s/1q473YIt2b18RqpOT8rdY6g 提取码: nkit

# License
The codes are released under the MIT License.

# References
* [EAST:An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2)

* [CTPN:Detecting Text in Natural Image with Connectionist Text Proposal Network](https://arxiv.org/abs/1609.03605)

* [Deep Matching Prior Network: Toward Tighter Multi-oriented Text Detection](https://arxiv.org/abs/1703.01425)
