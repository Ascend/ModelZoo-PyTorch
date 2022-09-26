# YoloV3模型使用说明

## Requirements
* NPU配套的run包安装
* Python 3.7.5
* numpy 1.20.2
* PyTorch(NPU版本)
* apex(NPU版本)
* (可选)参考《Pytorch 网络模型移植&训练指南》6.4.2章节，配置cpu为性能模式，以达到模型最佳性能；不开启不影响功能。

## Dataset Prepare
1. 下载COCO数据集
2. 新建文件夹data
3. 将coco数据集放于data目录下

## 预训练模型下载
1. 参考mmdetection/configs/yolo/README.md,下载对应预训练模型
2. 若无法自动下载，可手动下载模型，并放到/root/.cache/torch/checkpoints/文件夹下。

### Build MMCV

#### MMCV full version with CPU
```
cd ../
git clone -b v1.2.7 --depth=1 https://github.com/open-mmlab/mmcv.git

export MMCV_WITH_OPS=1
export MAX_JOBS=8
source ./test/env_npu.sh

cd mmcv
python3.7 setup.py build_ext
python3.7 setup.py develop
pip3.7 list | grep mmcv
```

#### Modified MMCV
将mmcv_need目录下的文件替换到mmcv的安装目录下。
安装完mmdet后执行以下命令：
```
/bin/cp -f mmcv_need/_functions.py ../mmcv/mmcv/parallel/
/bin/cp -f mmcv_need/builder.py ../mmcv/mmcv/runner/optimizer/
/bin/cp -f mmcv_need/data_parallel.py ../mmcv/mmcv/parallel/
/bin/cp -f mmcv_need/dist_utils.py ../mmcv/mmcv/runner/
/bin/cp -f mmcv_need/distributed.py ../mmcv/mmcv/parallel/
/bin/cp -f mmcv_need/optimizer.py ../mmcv/mmcv/runner/hooks/
```

###或者运行env_set.sh脚本，进行MMCV和mmdet的安装
```
bash env_set.sh
```

### Build MMDET from source
1. 下载modelzoo项目zip文件并解压
2. 压缩modelzoo\built-in\PyTorch\Official\cv\image_object_detection\YoloV3_for_PyTorch目录
3. 于npu服务器解压YoloV3_for_PyTorch压缩包
4. 执行以下命令，安装mmdet
```
cd YoloV3_for_PyTorch
pip3.7 install -r requirements/build.txt
pip3.7 install -v -e .
pip3.7 list | grep mm
```


## Train MODEL

### 导入环境变量(若安装mmcv时已导入，这步可跳过)
```
source ./test/env_npu.sh
```

### 单卡
1. 运行 train_1p.sh
```
chmod +x ./tools/dist_train.sh
sh train_1p.sh
```

### 8卡
1. 运行 train_8p.sh
```
chmod +x ./tools/dist_train.sh
sh train_8p.sh
```

## hipcc检查问题
若在训练模型时，有报"which: no hipcc in (/usr/local/sbin:..." 的日志打印问题，
而hipcc是amd和nvidia平台需要的，npu并不需要。
建议在torch/utils/cpp_extension.py文件中修改代码，当检查hipcc时，抑制输出。
将 hipcc = subprocess.check_output(['which', 'hipcc']).decode().rstrip('\r\n')修改为
hipcc = subprocess.check_output(['which', 'hipcc'], stderr=subporcess.DEVNULL).decode().rstrip('\r\n')

## invalid pointer问题
在Ubuntu、x86服务器上训练模型，有时会报invalid pointer的错误。
解决方法：去掉scikit-image这个依赖，pip3 uninstall scikit-image

## 单卡训练时，如何指定使用第几张卡进行训练
1. 修改 tools/train.py脚本
 将133行，cfg.npu_ids = range(world_size) 注释掉
 同时在meta['exp_name'] = osp.basename(args.config)后添加如下一行
 torch.npu.set_device(args.npu_ids[0])
2. 修改train_1p.sh
在PORT=29500 ./tools/dist_train.sh configs/yolo/yolov3_d53_320_273e_coco.py 1 --cfg-options optimizer.lr=0.001 --seed 0 --local_rank 0 后增加一个配置参数
--npu_ids k （k即为指定的第几张卡）

## 报No module named 'mmcv._ext'问题
在宿主机上训练模型，有时会报No module named 'mmcv._ext'问题，或者别的带有mmcv的报错。
解决方法：这一般是因为宿主机上安装了多个版本的mmcv，而训练脚本调用到了不匹配yolov3模型使用的mmcv，因此报mmcv的错误。
为了解决这个问题，建议在启动训练脚本前，先导入已经安装的符合yolov3模型需要的mmcv路径的环境变量。
export PYTHONPATH=mmcv的路径:$PYTHONPATH



