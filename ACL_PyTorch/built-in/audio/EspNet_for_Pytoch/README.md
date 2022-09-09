# Espnet_conformer模型PyTorch离线推理指导

## 1 环境准备 

1. 安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3.7 install -r requirements.txt  
```

2. 安装acl_infer，https://gitee.com/peng-ao/pyacl

3. 获取，修改与安装开源模型代码  

   EspNet安装比较复杂，请按照https://espnet.github.io/espnet/installation.html安装指导安装

   ```
   cd espnet
   git checkout v0.10.5
   ```

4. 下载网络权重文件

   下载路径：https://github.com/espnet/espnet/blob/master/egs/aishell/asr1/RESULTS.md

   对应Conformer(kernel size = 15) + SpecAugment + LM weight = 0.0下面的model link即可

   解压，将对应的conf，data, exp文件夹置于espnet/egs/aishell/asr1

5. 数据集下载：

   在espnet/egs/aishell/asr1/文件夹下运行bash run.sh --stage -1 –stop_stage -1下载数据集

   运行bash run.sh --stage 0 --stop_stage 0处理数据集

   运行bash run.sh --stage 1 --stop_stage 1处理数据集

   运行bash run.sh --stage 2 --stop_stage 2处理数据集

   运行bash run.sh --stage 3 --stop_stage 3处理数据集

6. 导出onnx，生成om离线文件

   ①静态shape

   将export_onnx.diff放在espnet根目录下，

   ```
   patch -p1 < export_onnx.diff
   cd ./egs/aishell/asr1/
   bash export_onnx.sh
   ```

   生成encoder.onnx，运行python3.7.5 adaptespnet.py生成encoder_revise.onnx

   ②动态shape

   将export_onnx_dynamic.diff放在espnet根目录下，运行脚本生成encoder.onnx

   ```
   patch -p1 < export_onnx_dynamic.diff
   cd ./egs/aishell/asr1/
   bash export_onnx.sh
   ```

7. 运行encoder.sh生成离线om模型， encoder_262_1478.om

   ${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

   ```
   bash encoder.sh Ascend${chip_name} # Ascend310P3
   ```


## 2 离线推理 

   首先为了获得更好的性能，可以首先设置日志等级，商发版本默认ERROR级别

```
export ASCEND_GLOBAL_LOG_LEVEL=3
/usr/local/Ascend/driver/tools/msnpureport -g error -d 0
```

1.  拷贝encoder_out_shape.json文件到espnet/egs/aishell/asr1目录下

2. 获取精度

   ①静态shape

   首先修改acc.diff文件中的om模型路径（约162行）为生成的om路径

   ```
   cd espnet
   patch -p1 < acc.diff
   cd espnet/egs/aishell/asr1
   bash acc.sh
   ```

   ②动态shape

   首先修改acc_dynamic.diff文件中的om模型路径（约162行）为生成的om路径

   ```
   cd espnet
   patch -p1 < acc_dynamic.diff
   cd espnet/egs/aishell/asr1
   bash acc.sh
   ```

   即可打屏获取精度，精度保存在文件espnet/egs/aishell/asr1/exp/train_sp_pytorch_train_pytorch_conformer_kernel15_specaug/decode_test_decode_lm0.0_lm_4/result.txt

3. 获取性能

   需要首先配置环境变量:

   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

   运行脚本infer_perf.py

   ```
   python3.7.5 infer_perf.py即可获取打印的fps性能
   ```

   

|       模型       | 官网pth精度 | 310P离线推理精度 | gpu性能 | 310P性能 |
| :--------------: | :---------: | :-------------: | :-----: | :-----: |
| espnet_conformer |    5.1%     |      分档5.4%；动态：5.1%      | 261fps  | 分档：430fps；动态：25fps |
