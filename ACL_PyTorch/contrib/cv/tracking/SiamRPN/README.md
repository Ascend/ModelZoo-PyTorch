# SiamRPN模型PyTorch推理指导

## 1 环境准备
1. 获取开源代码仓
- 得到本项目代码后，将 SiamRPN 项目放置在/home目录下,进入/home/SiamRPN目录下，下载开源代码仓
```
git clone https://github.com/STVIR/pysot.git 
```
  
- 确认获取的开源 pysot 项目文件存放在 /home/SiamRPN 目录下，进入 /home/SiamRPN/pysot 目录下执行
```
patch -N -p1 < ../SiamRPN.patch
```

2. 获取数据集  
- 将数据集VOT2016下载并放在 /root/datasets 目录下
```
wget -P /root/datasets https://ascend-pytorch-one-datasets.obs.cn-north-4.myhuaweicloud.com/train/zip/VOT2016.zip
cd //
cd /root/datasets
unzip VOT2016.zip
rm -rf VOT2016.zip
```
- （备注：将获取的 VOT2016 数据集文件放在 /root/datasets 目录下）


3. 安装依赖
- 进入 /home/SiamRPN 目录下
```shell
cd //
cd /home/SiamRPN
pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```
- （备注：若有安装失败的请单独重新安装,若有需要,也可使用conda指令安装）

4. 获取pth权重文件

```
wget -P /home/SiamRPN/pysot/experiments/siamrpn_r50_l234_dwxcorr https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/tracking/SiamRPN/model.pth
```

- (备注:将 model.pth 权重文件要放在 ../siamrpn_r50_l234_dwxcorr 目录下)

5. 运行setup.py
- 进入 /home/SiamRPN/pysot 目录下执行
```
export PYTHONPATH=/home/SiamRPN/pysot:$PYTHONPATH
python setup.py build_ext --inplace install
```



## 2 离线推理

310上执行，执行时使用 npu-smi info 查看设备状态，确保device空闲

```shell
# (j进入 /home/SiamRPN 下执行)
# 转成onnx
bash test/pth2onnx.sh
# 转成om
bash test/onnx2om.sh
# 进行评估
bash test/eval_acc_perf.sh
```


- 评测结果：

- 310精度
```
------------------------------------------------------------
|Tracker Name| Accuracy | Robustness |   Average   |  EAO  |
------------------------------------------------------------
|  VOT2016   |  0.639   |   0.177    |    42fps    | 0.483 |
------------------------------------------------------------
```

- 参考pth精度
```
------------------------------------------------------------
|Tracker Name| Accuracy | Robustness |   Average   |  EAO  |
------------------------------------------------------------
|  VOT2016   |  0.642   |   0.196    |    35fps    | 0.464 |
------------------------------------------------------------
```

  
- 性能计算方式： 
  fps计算方式为单位时间内处理的图片数量，即 图片数量 / 时间 。
  根据310单device需乘以4之后再和T4对比，故310单卡性能理论计算为42×4=168fps。

- 备注：
- (1) 310精度相较于T4下降0.3%，但鲁棒性和EAO均有提升。310单device的实际平均性能为42fps。T4单卡平均性能为35fps，由于运行场景等干扰因素不同，会导致结果有所浮动，35fps为多次测量后平均近似值，供参考。
- (2) 性能数据(speed)在推理过程中会展示，在推理结束后会展示平均性能(average speed)。
- (3) 本推理为视频追踪，输入对象为视频，故不设置多batch。 

