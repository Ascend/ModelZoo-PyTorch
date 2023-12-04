## 构建虚拟环境

`conda create --name vgg16 python=3.9`
激活：`conda activate vgg16`

## 安装依赖

`pip3 install -r requirements.txt`

编译pt插件，在dist目录下安装torh_aie

## 下载pth模型

自行下载模型pth文件并放置在`vgg16`路径下
链接：https://gitee.com/link?target=https%3A%2F%2Fdownload.pytorch.org%2Fmodels%2Fvgg16-397923af.pth

## trace得到ts文件

将model_path改为自己tar模型的路径
`python3 export.py --model_path=./vgg16-397923af.pth`

## 模型推理 - 获取精度

将data_path改为自己目录下数据集label的路径
`python3 run.py --data_path /home/pttest_models/datasets/ImageNet_50000/val`

## 推理性能 - ts

将--ts_path改为自己目录下的ts路径
`python3 perf_right_one.py --mode=ts --ts_path=./shufflenetv1.ts`