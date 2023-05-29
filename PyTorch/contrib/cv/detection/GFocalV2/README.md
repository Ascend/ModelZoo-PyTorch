# GFocalV2

This implements training of GFocalV2 on the Coco dataset, mainly modified from [pytorch/examples](https://github.com/open-mmlab/mmdetection).

## GFocalV2 Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.
Therefore, GFocalV2 model need to be modified in the following aspects:

1. Converting tensors with the dynamic shapes into tensors with fixed shapes. (This is the hardest one)
2. Several operations, like the sum of `INT64`, are not supported on the NPU, so we modified tensors' `dtype` when needed
3. Framework bottlenecks lead to poor performance, so we improve the original code to improve the performance of the model
4. We used Apex for mmdtection due to the hardware defects of the NPU
5. ...


## Requirements

- NPU配套的run包安装
- Python 3.7.5
- PyTorch(NPU版本)
- apex(NPU版本)
- MMCV v1.2.7
### Document and data preparation
1. 下载压缩modelzoo\contrib\PyTorch\cv\image_object_detection\GFocalV2 文件夹
2. 于npu服务器解压GFocalV2压缩包
3. 下载coco2017数据集
4. 将coco数据集放于GFocalV2/data目录下，目录结构如下：
```
GFocalV2
├── configs
├── data
│   ├── coco
│       ├── annotations   796M
│       ├── train2017     19G
│       ├── val2017       788M
```
### Download and modify mmcv

1. 在GFocalV2目录上级目录，下载mmcv，最好是1.2.7版本的（版本要求是1.2.5以上，1.3.0以下）
```
git clone -b v1.2.7 --depth=1 https://github.com/open-mmlab/mmcv.git
```
2. 进入GFocalV2目录，用mmcv_need里的文件替换mmcv中对应的文件
```
cp -f ./mmcv_need/_functions.py ./mmcv/mmcv/parallel/
cp -f ./mmcv_need/builder.py ./mmcv/mmcv/runner/optimizer/
cp -f ./mmcv_need/distributed.py ./mmcv/mmcv/parallel/
cp -f ./mmcv_need/data_parallel.py ./mmcv/mmcv/parallel/
cp -f ./mmcv_need/dist_utils.py ./mmcv/mmcv/runner/
cp -f ./mmcv_need/optimizer.py ./mmcv/mmcv/runner/hooks/
```
3. 以下三个文件的替换是为了在log中打印出FPS的信息
```
cp -f ./mmcv_need/iter_timer.py ../mmcv/mmcv/runner/hooks/
cp -f ./mmcv_need/base_runner.py ../mmcv/mmcv/runner/
cp -f ./mmcv_need/epoch_based_runner.py ../mmcv/mmcv/runner/
```
### Configure the environment
```
先source环境变量
source GFocalV2/test/env_npu.sh  
```
1. 配置安装mmcv
```
cd mmcv
MMCV_WITH_OPS=1 pip3.7 install -e .
cd ..
pip3 list | grep mmcv  # 查看版本和路径
``` 
2. 配置安装mmdet
```
cd GFocalV2
pip3 install -r requirements/build.txt
python3 setup.py develop
cd ..
```
3. 修改apex中的113行，主要是为了支持O1，参考路径root/archiconda3/envs/fcos/lib/python3.7/site-packages/apex/amp/utils.py
```
if cached_x.grad_fn.next_functions[1][0].variable is not x:
```
改成
```
if cached_x.grad_fn.next_functions[0][0].variable is not x:
```
## Train MODEL
进入GFocalV2目录下
### 1p
导入环境变量，修改train_full_1p.sh权限并运行
```
chmod +x ./test/train_full_1p.sh
bash ./test/train_full_1p.sh --data_path=./data/coco
```

### 8p
导入环境变量，修改train_full_8p.sh权限并运行
```chmod +x ./test/train_full_8p.sh
bash ./test/train_full_8p.sh --data_path=./data/coco
```

### Eval
修改train_eval_1p.sh权限并运行
```
chmod +x ./test/train_eval_1p.sh
bash ./test/train_eval_1p.sh --data_path=./data/coco
```
### finetuning
修改train_finetune_1p.sh权限并运行
```
chmod +x ./test/train_eval_1p.sh
bash ./test/train_finetune_1p.sh --data_path=./data/coco --checkpoint=xxx（可选，gfocal模型的权重文件）
```
### Demo

```
source ./test/env_npu.sh
python3 demo.py --checkpoint xxx(可选，gfocal模型的权重文件，默认./work_dirs/gfocal_r50_fpn_1x/latest.pth） --img xxx(可选，测试图片）
```
## GFocalV2 training result 

| Acc@1    | FPS       | Npu/Gpu_nums | Epochs   | AMP_Type | Loss_Scale |
| :------: | :------:  | :------:     | :------: | :------: | :------:   |
| 23.2     | 14.6      | 1p Gpu       | 1        | O1       | 128.0    |
| 41.0     | 71.3     | 8p Gpu       | 12       | O1       | 128.0    |
| 23.5     | 3.46       | 1p Npu       | 1        | O1       | 128.0       |
| 40.9     | 27.15      | 8p Npu       | 12       | O1       | 128.0       |
