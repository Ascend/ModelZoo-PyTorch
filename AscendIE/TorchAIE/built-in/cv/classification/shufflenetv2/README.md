## 构建虚拟环境

`conda create --name shufflenetv2 python=3.9`
激活：`conda activate shufflenetv2`

## 安装依赖

`pip3 install -r requirements.txt`

编译pt插件，在dist目录下安装torh_aie

## 下载pth模型

自行下载模型pth文件并放置在`shufflenetv2`路径下
链接：https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth

## trace得到ts文件

将model_path改为自己模型的路径
`python3 export.py --model_path=./shufflenetv2_x1-5666bf0f80.pth`

## 模型推理 - 获取精度

将data_path改为自己目录下数据集的路径
`python3 run.py --data_path ./datasets/ImageNet_50000/val`

## 推理性能 - ts

将--ts_path改为自己目录下的ts路径
`python3 perf.py --mode=ts --batch_size=1`