# Fcos

This implements training of Fcos on the Coco dataset, mainly modified from [pytorch/examples](https://github.com/open-mmlab/mmdetection).

## Fcos Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 
Therefore, Fcos is re-implemented by changing mmdet and mmcv .


## Requirements

- NPU配套的run包安装
- Python 3.7.5
- PyTorch(NPU版本)
- apex(NPU版本)

### Document and data preparation
1. 下载压缩modelzoo\contrib\PyTorch\cv\image_object_detection\Fcos文件夹
2. 于npu服务器解压Fcos压缩包
3. 下载coco数据集
4. 将coco数据集放于Fcos/data目录下

### Download and modify mmcv
1. 下载mmcv，最好是1.2.7版本的（版本要求是1.2.5以上，1.3.0以下）
```
git clone -b v1.2.7 git://github.com/open-mmlab/mmcv.git
```
2. 用mmcv_need里的文件替换mmcv中对应的文件
```
cp -f mmcv_need/_functions.py ../mmcv/mmcv/parallel/
cp -f mmcv_need/builder.py ../mmcv/mmcv/runner/optimizer/
cp -f mmcv_need/distributed.py ../mmcv/mmcv/parallel/
cp -f mmcv_need/data_parallel.py ../mmcv/mmcv/parallel/
cp -f mmcv_need/dist_utils.py ../mmcv/mmcv/runner/
cp -f mmcv_need/optimizer.py ../mmcv/mmcv/runner/hooks/
cp -f mmcv_need/checkpoint.py ../mmcv/mmcv/runner/
```
3. 以下三个文件的替换是为了在log中打印出FPS的信息，替换与否对模型训练无影响
```
cp -f mmcv_need/iter_timer.py ../mmcv/mmcv/runner/hooks/
cp -f mmcv_need/base_runner.py ../mmcv/mmcv/runner/
cp -f mmcv_need/epoch_based_runner.py ../mmcv/mmcv/runner/
```
### Configure the environment
1. 推荐使用conda管理
```
conda create -n fcos --clone env  # 复制一个已经包含依赖包的环境 
conda activate fcos
```
2. 配置安装mmcv
```
cd mmcv
export MMCV_WITH_OPS=1
export MAX_JOBS=8
python3.7 setup.py build_ext
python3.7 setup.py develop
pip3 list | grep mmcv  # 查看版本和路径
``` 
3. 配置安装mmdet
```
cd Fcos
pip3 install -r requirements/build.txt
python3.7 setup.py develop
pip3 list | grep mmdet  # 查看版本和路径
```
4. 修改apex中的113行，主要是为了支持O1，参考路径root/archiconda3/envs/fcos/lib/python3.7/site-packages/apex/amp/utils.py
```
if cached_x.grad_fn.next_functions[1][0].variable is not x:
```
改成
```
if cached_x.grad_fn.next_functions[0][0].variable is not x:
```
## Train MODEL

### 进入Fcos文件夹下
```
cd FCOS
```

### 1p

```

bash ./test/train_full_1p.sh  --data_path=数据集路径
```

### 8p

```
bash ./test/train_full_8p.sh  --data_path=数据集路径
```

### 多机多卡性能数据获取流程

```shell
	1. 安装环境
	2. 开始训练，每个机器所请按下面提示进行配置
        bash ./test/train_performance_multinodes.sh  --data_path=数据集路径 --batch_size=单卡batch_size --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
```

### Eval
修改eval.sh权限并运行
```
chmod +x ./scripts/eval.sh
bash ./scripts/eval.sh
```

### 单p推理
1. 运行demo.py
```
python3.7 demo.py xxx.pth
```


### 导出onnx
1. 下载mmdetection2.11，在文件夹下重新编译
```
git clone -b v2.11.0 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
python3.7 setup.py develop
```
2. 运行pthtar2onx.py
```
python3.7 pthtar2onx.py
```


## Fcos training result 

| Acc@1    | FPS       | Npu/Gpu_nums | Epochs   | AMP_Type | Loss_Scale |
| :------: | :------:  | :------:     | :------: | :------: | :------:   |
| 12.6     | 19.2      | 1p Gpu       | 1        | O1       | dynamic    |
| 36.2     | 102.0     | 8p Gpu       | 12       | O1       | dynamic    |
| 16.4     | 6.8       | 1p Npu       | 1        | O1       | 32.0       |
| 36.2     | 19.4      | 8p Npu       | 12       | O1       | 32.0       |
