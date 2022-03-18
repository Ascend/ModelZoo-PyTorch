# Shape Robust Text Detection with Progressive Scale Expansion Network

## Requirements
* Python 3.7.5
* PyTorch
* pyclipper
* Polygon3
* opencv-python 3.4

## test
1. edit test_npu.sh data_dir(数据集目录) resume(待测试模型文件) output_file(输出文件名)

2. edit eval/eval_ic.sh s(测试脚本输出zip文件)

3. run sh test_npu.sh

4. run cd eval;  sh eval_ic.sh

```
python test_npu.py \
	--long_size 2240 \
	--npu 1\
	--resume "/PATH/TO/CONFIGED/deploy/PSENet/8p/best/npu8pbatch64lr4_0.3401_0.9416_0.8407_0.9017_521.pth"\ #修改待测试模型文件
	--data_dir '/PATH/TO/CONFIGED/data/ICDAR/Challenge/' \ #修改数据集目录
	--output_file 'npu8p64r4521' #修改输出文件名
```

```
python script.py -g=gt.zip -s=../../outputs/npu8p64r4521.zip #s为测试脚本输出zip文件
```
