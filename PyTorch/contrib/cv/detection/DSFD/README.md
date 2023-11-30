## DSFD: Dual Shot Face Detector

[ Dual Shot Face Detector](https://arxiv.org/abs/1810.10220?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+(ExcitingAds!+cs+updates+on+arXiv.org))

### 1、Description

DSFD for face detection in general scene with high detection rate

### 2、Prepare data

1、download [WIDER Face Datese](http://shuoyang1213.me/WIDERFACE/)t， locate /opt/npu/

2、data process perform the following create data for train

```
python prepare_wider_data
```

### 3、Train

    Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
1、pretrained weight

download [pretrained weights](链接：https://pan.baidu.com/s/1qbQsOcgD3vuJ5m3Jnu6HTw  提取码：vbo9)

2、train full1p get train_1p  log 

```
bash ./test/train_full_1p.sh
```

3、train 1p performance

```
bash./test/train_performance_1p.sh
```

4、train full 8p get train_8p log 

```
bash ./test/train_full_8p.sh
```

5、train 8p performance

```
bash ./test/train_performance_8p.sh 
```

6、use resume training

```
#enable resume training
add --resume "path/to/checkpoint" to .sh
```

### 4、Data inference

```
1.download wider_face_test.mat and wider_face_val.mat in /tools/infer_tools
2.cd /tools/infer_tools
python wider_face_test.py
```

### 5、Evalution

1、do setup first

```
cd /tools/eval_tools
python setup.py build_ext --inplace
```

2、download ground_truth and unzip to /tools/eval_tools

3、get data evaluation result

```
python evaluation.py
```

### 6、Demo

```python
python demo.py --network 'resnet152'
```



### 7、DSFD training result

| Acc                        | Npu_numbers | epochs | AMP_type |
| -------------------------- | ----------- | ------ | -------- |
| -                          | 1           | 1      | O2       |
| E:0.9368 M:0.9282 H:0.8460 | 8           | 100    | O2       |

Reference:

|                 | Acc                          |
| :-------------- | ---------------------------- |
| 参考精度        | E:0.951 M:0.936 H:0.837      |
| GPU 8P 自测精度 | E 0.9473, M 0.9362, H 0.8651 |


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md