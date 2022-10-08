###  Ernie3模型PyTorch离线推理指导

### 1. 环境准备

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 第三方依赖

   ```
   git clone https://gitee.com/peng-ao/pyacl.git
   cd pyacl
   pip3 install .
   ```

   若有其他需要安装的包，请自行安装

3. [获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)

   将benchmark.x86_64或benchmark.aarch64放到当前Ernie3_for_Pytorch目录下


### 2. 离线推理

310P上执行，执行时使npu-smi info查看设备状态，确保device空闲

${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_GLOBAL_LOG_LEVEL=3
/usr/local/Ascend/driver/tools/msnpureport -g error -d 0
```

导出onnx:

```bash
python3 export_onnx.py --model_name ernie-3.0-base-zh --model_type AutoModelForSequenceClassification --save_path ernie/
paddle2onnx --model_dir ernie/ --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file model.onnx --opset_version 11
```

获取模型性能和精度，保存在results.txt文件中

```bash
bash run.sh ernie-3.0-base-zh 8 128 Ascend${chip_name} 0 csl
```

 run.sh的参数分别为使用的模型名称，batch_size, max_seq_length, 芯片类型， 芯片device_id, 任务类型

