## 构建虚拟环境

`conda create --name mobilenetV1 python=3.9`
激活：`conda activate mobilenetV1`

## 安装依赖

`pip3 install -r requirements.txt`

编译pt插件，在dist目录下安装torh_aie

## 下载pth模型

自行下载模型tar文件并放置在`stargan`路径下
链接：https://gitee.com/link?target=https%3A%2F%2Fascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com%2Fmodel%2F1_PyTorch_PTH%2FStarGan%2FPTH%2F200000-G.pth

## trace得到ts文件

将`input_file`改为自己tar模型的路径
`python3 export.py --input_file=/onnx/stargan/200000-G.pth`

## 模型推理 - 获取精度

将`attr_path`改为自己目录下数据集label的路径，将`celeba_image_dir`改为自己目录下数据集image的路径

`python3 StarGAN_pre_processing.py --result_dir './result_baseline' --attr_path '/onnx/dataset/celeba/list_attr_celeba.txt' --celeba_image_dir '/onnx/dataset/celeba/images'  --batch_size 16 --ts_model_path "./stargan.ts"`

## 推理性能 - ts

将--ts_path改为自己目录下的ts路径
`python3 perf.py --mode ts --batch-size 1`