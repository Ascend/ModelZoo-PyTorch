# TDNN模型pytorch离线推理指导

## 1 环境准备

1.获取，修改与安装开源模型代码

```shell
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain    
git checkout  develop    
git reset --hard 51a2becdcf3a337578a9307a0b2fc3906bf20391
export PYTHONPATH=`pwd`:$PYTHONPATH
cd ..
git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
cd MagicONNX && git checkout 8d62ae9dde478f35bece4b3d04eef573448411c9
pip install .
```
将源码包中文件放入speechbrain/templates/speaker_id中
```shell
cd speechbrain
git apply --reject --whitespace=fix templates/speaker_id/modify.patch
```

2.获取权重文件

https://www.hiascend.com/zh/software/modelzoo/detail/1/f4f4103245624c1a8637f8a5eadd950c
将模型权重文件夹best_model放入speechbrain/templates/speaker_id下，将hyperparams.yaml文件放入best_model中

3.获取数据集

预处理阶段自动下载
```shell
python3 tdnn_preprocess.py
```

## 2 模型转换
```shell
# 生成tdnn_bs64.onnx
python3 tdnn_pth2onnx.py 64
# 优化onnx模型
python3 -m onnxsim tdnn_bs64.onnx tdnn_bs64s.onnx
python3 modify_onnx.py tdnn_bs64s.onnx
# 生成om模型
bash atc.sh tdnn_bs64s.onnx
```

## 3 离线推理

```shell
bash om_infer.sh 64
python3 tdnn_postprocess.py
```
**评测结果：**

由于TensorRT不支持原模型，故只能对比修改后的模型性能。
| 模型              | pth精度        | 710离线推理精度      | 基准性能      | 710性能  |
| :------:          | :------:       | :------:            | :------:     | :------: |
| TDNN bs64         | 99.93%         | 99.93%              | -            |  2467fps  |
| TDNN修改 bs64     | -              | -                   | 2345.179 fps |  3815.886fps  |