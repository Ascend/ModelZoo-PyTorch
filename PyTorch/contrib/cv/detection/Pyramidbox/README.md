## Pyramidbox

​	A Context-assisted Single Shot Face Detector.



### Requirements

```
根据requirements.txt安装依赖环境
- Download the WIDER_FACE dataset from http://shuoyang1213.me/WIDERFACE/
```



在代码仓下创建WIDER_FACE目录，存放数据集文件

```
|-WIDER_FACE
	|-wider_face_split
		|-wider_face_test.mat
		|-wider_face_test_filelist.txt
		|-wider_face_train.mat
		|-wider_face_train_bbx_gt.txt
		|-wider_face_val.mat
		|-wider_face_val_bbx_gt.txt
	|-WIDER_train
		|-images
			|-0--Parade
			|-1--Handshaking
			...
	|-WIDER_val
		|-images
			|-0--Parade
			|-1--Handshaking
			...
```



参照原代码仓README下载vgg权重，放在weights目录下

```
|-Pyramidbox
	|-weights
		|-vgg16_reducedfc.pth
```



### prepare

运行prepare_wider_data.py

```
python prepare_wider_data.py --data_path='数据集路径'
```



### Training

单卡训练

```
python train.py --batch_size=8 --lr=5e-4 
```

多卡训练

```
python -m torch.distributed.launch --nproc_per_node=8 train.py  --world_size=8 --batch_size=8 --lr=5e-4 --multinpu=True --device_list='0,1,2,3,4,5,6,7' 
```



### Test

在运行wider_test.py前，应先做以下修改：

```
1、修改第53行
sys.path.append("/home/wch/Pyramidbox/") #根据代码仓实际所在位置进行修改
```

修改后，运行wider_test.py

```
python tools/wider_test.py --model="/home/wch/Pyramidbox/weights/pyramidbox.pth" --data_path='数据集路径'
#model参数根据模型权重文件保存位置进行修改
```

运行以下脚本，评估精度

```
cd evaluate
python setup.py build_ext --inplace
python evaluation.py --pred ../output/pyramidbox1_val/ --gt '数据集路径/wider_face_split'
```



### 启动脚本

8卡训练，并显示性能和精度

```
bash ./test/train_full_8p.sh --data_path='数据集路径'
```

测试单卡训练性能

```
bash ./test/train_performance_1p.sh --data_path='数据集路径'
```

测试多卡训练性能

```
bash ./test/train_performance_8p.sh --data_path='数据集路径'
```

模型迁移脚本,注意脚本中的resume参数只能指定为保存的“pyramidbox_checkpoint.pth”权重

```
bash ./test/train_finetune_1p.sh --data_path='数据集路径'
```

精度数据

```
==================== Results ====================
Easy   Val AP: 0.9519612346942784
Medium Val AP: 0.9446576258551937
Hard   Val AP: 0.9053749943031708
=================================================
```

