 **介绍**
Ascend预置数据集和Python第三方库

 **一、预置数据集**

Shell脚本中，访问时用提供变量即可

| 领域  | 数据集名称        | 访问变量            |
|-----|--------------|---------------|
| CV  | IMG12_train | $IMG12_train |
| CV  | IMG12_val | $IMG12_val |
| CV  | CIFAR10      | $CIFAR10      |
| CV  | ADEChallenge      | $ADEChallenge      |
| CV  | COCO_train      | $COCO_train      |
| CV  | COCO_val      | $COCO_val      |
| CV  | SBD      | $SBD      |
| CV  | VOC2012      | $VOC2012      |
| CV  | Cityscape      | $cityscapes_dataset   |
| CV  | icdar2015      | $icdar2015_train   |
| NLP | Wikipedia_CN | $Wikipedia_CN |
| NLP | WMT_ENDE     | $WMT_ENDE      |
| NLP | SQUADV2      | $SQUADV2       |

使用示例：python3 run.py --dataurl=$cityscapes_dataset ...

预置数据集持续建设中，若有新增需求，请提交ISSUE，标题注明[新增数据集]，内容写上数据集名称和下载地址，\
涉及数据集由第三方开发者提供，仅用于性能或精度调试or自测试，华为方不会存储或使用该数据集。

 **二、预训练模型**
| 领域  | 数据集名称        | 访问变量            |
|-----|--------------|---------------|
| CV  | resnetv1_50_ckpt| $resnetv1_50_ckpt|


 **三、Python第三方库**

安装第三方库依赖使用"pip3"、"pip3.7"，已安装的库：
```
Package               Version
--------------------- --------
absl-py               0.11.0
anyconfig             0.9.11
astor                 0.8.1
cached-property       1.5.2
cycler                0.10.0
decorator             4.4.2
easydict              1.9
enum34                1.1.10
gast                  0.2.2
google-pasta          0.2.0
grpcio                1.33.2
h5py                  3.1.0
hccl                  0.1.0
imageio               2.9.0
imgaug                0.4.0
importlib-metadata    2.0.0
Keras                 2.3.1
Keras-Applications    1.0.8
Keras-Preprocessing   1.1.2
kiwisolver            1.3.1
Markdown              3.3.3
matplotlib            3.3.3
mpmath                1.1.0
munch                 2.5.0
networkx              2.5
npu-bridge            1.15.0
numpy                 1.19.4
opencv-contrib-python 4.4.0.46
opencv-python         4.4.0.46
opt-einsum            3.3.0
Pillow                8.0.1
pip                   20.3
protobuf              3.14.0
pyclipper             1.2.0
pyparsing             2.4.7
python-dateutil       2.8.1
PyWavelets            1.1.1
PyYAML                5.3.1
scikit-image          0.17.2
scipy                 1.5.4
setuptools            41.2.0
Shapely               1.7.1
six                   1.15.0
sympy                 1.6.2
te                    0.4.0
tensorboard           1.15.0
tensorflow            1.15.0
tensorflow-estimator  1.15.1
termcolor             1.1.0
tifffile              2020.11.26
topi                  0.4.0
tqdm                  4.54.0
utils                 1.0.1
Werkzeug              1.0.1
wheel                 0.35.1
wrapt                 1.12.1
zipp                  3.4.0
```
 **四、从个人的OBS桶中下载**

如果需要的文件大小<500M，可以使用obsutil命令下载：
```
obsutil cp obs://obsxxx/xxx/xxx.pb ./model/ -f -r
```
