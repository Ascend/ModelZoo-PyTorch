# Albert

base on [Albert-base-v2](https://github.com/lonePatient/albert_pytorch)

## 运行说明
- 下载[原始代码仓](https://github.com/lonePatient/albert_pytorch)
```
git clone https://github.com/lonePatient/albert_pytorch.git
cd albert_pytorch
git checkout 46de9ec
git apply ../0001-init.patch
cd ../
```
- 下载[数据集](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip) 并解压albert_pytorch/dataset/SST-2。
- 下载[预训练模型](https://drive.google.com/open?id=1byZQmWDgyhrLpj8oXtxBG6AA52c8IHE- ) 并解压到albert_pytorch/prev_trained_model/albert_base_v2。
- 下载[训练好的模型](https://pan.baidu.com/s/1G5QSVnr2c1eZkDBo1W-uRA )（提取码：mehp ）并解压到albert_pytorch/outputs/SST-2。
- 文件夹若无请新建
- `pip install -r requirements.txt`
- 获取 benchmark 工具

```bash
#设置环境
source env.sh

#pth2onnx, 生成的 onnx om 文件在 outputs 文件夹下。
bash ./test/pth2om.sh 

# 前处理，pth_dir参数是预训练模型的路径
python3.7 Albert_preprocess.py --pth_dir=./albert_pytorch/outputs/SST-2/
# 生成bin文件，bin文件在bin_dir文件夹下，info和label文件在当前目录下
python3.7 gen_dataset_info.py --pth_dir=./albert_pytorch/outputs/SST-2/ --bin_dir=./bin/

#运行 om 模型，生成的 perf 文件在 result 文件夹下，fps和acc打印到屏幕
bash test/eval_acc_perf.sh

# 基准性能测试
bash perf_g.sh
```
## 结果

精度性能

| 模型      | pth精度  | 310精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| Albert bs1  | acc:0.928 | acc:0.927  |  276.61fps | 231.39fps | 
| Albert bs16 | acc:0.928  | acc:0.927 | 577.73fps | 300.83fps | 

