# DeeplabV3模型使用说明

## Requirements
* NPU配套的run包安装
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)
* mmcv-full 1.3.9

### Dataset Prepare
1. 下载cityscapes数据集

2. 新建文件夹data

3. 将cityscas数据集放于data目录下

   ```shell
   ln -s /path/to/cityscapes/ ./data
   ```

4. 处理数据集，`**labelTrainIds.png` 被用来训练

   ```shell
   python3 tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
   # python3 tools/convert_datasets/cityscapes.py /path/to/cityscapes --nproc 8
   ```

### 预训练模型下载
* 若无法自动下载，可手动下载resnet50_v1c.pth，并放到/root/.cache/torch/checkpoints/文件夹下。

### 脚本环境安装
#### 运行env_set.sh脚本，进行MMCV和mmsegmentation的安装
```shell
bash env_set.sh
```
编译mmcv耗时较长，请耐心等待

### 手动环境安装
#### Build MMSEG from source

1. 下载项目zip文件并解压
3. 于npu服务器解压DeeplabV3_for_PyTorch压缩包
4. 执行以下命令，安装mmsegmentation
```shell
cd DeeplabV3_for_PyTorch
pip3.7 install -r requirements.txt
pip3.7 install -e .
pip3.7 list | grep mm
```


#### Build MMCV

##### MMCV full version with cpu
```shell
source ./test/env_npu.sh
cd ..
git clone -b v1.3.9 --depth=1 https://github.com/open-mmlab/mmcv.git
export MMCV_WITH_OPS=1
export MAX_JOBS=8

cd mmcv
python3.7 setup.py build_ext
python3.7 setup.py develop
pip3.7 list | grep mmcv
# 安装opencv-python-headless, 规避cv2引入错误
pip3.7 uninstall opencv-python
pip3.7 install opencv-python-headless
```

##### Modified MMCV
将mmcv_need目录下的文件替换到mmcv的安装目录下。

```shell
cd ../DeeplabV3_for_PyTorch
/bin/cp -f mmcv_need/_functions.py ../mmcv/mmcv/parallel/
/bin/cp -f mmcv_need/scatter_gather.py ../mmcv/mmcv/parallel/
/bin/cp -f mmcv_need/dist_utils.py ../mmcv/mmcv/runner/
```

## Training

```shell
# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=real_data_path

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path
```

## 多机多卡性能数据获取流程

```shell
	1. 安装环境
	2. 开始训练，每个机器所请按下面提示进行配置
        bash ./test/train_performance_multinodes.sh  --data_path=数据集路径 --batch_size=单卡batch_size --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
```

## hipcc检查问题
若在训练模型时，有报"which: no hipcc in (/usr/local/sbin:..." 的日志打印问题，
而hipcc是amd和nvidia平台需要的，npu并不需要。
建议在torch/utils/cpp_extension.py文件中修改代码，当检查hipcc时，抑制输出。
将 hipcc = subprocess.check_output(['which', 'hipcc']).decode().rstrip('\r\n')修改为
hipcc = subprocess.check_output(['which', 'hipcc'], stderr=subporcess.DEVNULL).decode().rstrip('\r\n')

## 报No module named 'mmcv._ext'问题
在宿主机上训练模型，有时会报No module named 'mmcv._ext'问题(按照setup.py build_ext安装一般不会遇到此问题)，或者别的带有mmcv的报错。
解决方法：这一般是因为宿主机上安装了多个版本的mmcv，而训练脚本调用到了不匹配DeeplabV3模型使用的mmcv，因此报mmcv的错误。
为了解决这个问题，建议在启动训练脚本前，先导入已经安装的符合DeeplabV3模型需要的mmcv路径的环境变量。export PYTHONPATH=mmcv的路径:$PYTHONPATH
