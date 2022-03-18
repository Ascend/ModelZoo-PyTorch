### Install

进入到项目下面，执行下面的命令

```bash
pip install -r requirements.txt
pip install -v -e .
pip install cython 
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```



### Training

模型存在动态Shape，为提升性能，固定shape使用分档策略，故第一个epoch前期持续有算子编译，现象为iter_time抖动。性能数据请关注第一个epoch后期或第二个epoch之后。精度测试须300epoch，第一个epoch性能波动对整体影响较小。

shell脚本会将传入的`data_path`软连接到`./datasets/COCO`，默认只支持coco数据集，这里与原仓一致，使用自己的数据集需首先将数据转为coco格式。

训练完成会对最后一个epoch的权重做一次测试，该次精度测试结果仅作参考。原仓代码会对最后15个epoch的权重做测试（模型最后15个epoch关闭马赛克增强并增加l1loss），选取最高的MAP值，因此须在训练完成后单独执行测试脚本，测试最后15个epoch的所有权重取最高精度值，并将该权重保存为`best_ckpt.pth`。

```bash
# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path
```

### Evaluting

精度测试，须在训练完成后，单独执行eval脚本，`--pth_path`指定权重文件目录，该脚本评测最后15个epoch的权重文件，输出最好结果，并将该权重保存为`best_ckpt.pth`。

```bash
#test 8p accuracy
bash test/train_eval_1p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path
```

### Yolox-x Result

| 名称   | 精度 | 性能    |
| ------ | ---- | ------- |
| GPU-1p | -    | 20fps   |
| NPU-1p | -    | 20.5fps |
| GPU-8p | 50.7 | 106fps  |
| NPU-8p | 50.5 | 140fps  |

