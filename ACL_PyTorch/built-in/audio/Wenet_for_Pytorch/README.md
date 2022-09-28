# Wenet模型PyTorch离线推理指导

## 1 环境准备 

1. 安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3.7 install -r requirements.txt  
```

2. 改图工具om_gener

   ```
   git clone https://gitee.com/liurf_hw/om_gener.git
   cd om_gener
   pip3 install .
   ```

其他需要安装的请自行安装

3. 获取开源模型代码  

```
git clone https://github.com/wenet-e2e/wenet.git
cd wenet
git reset 9c4e305bcc24a06932f6a65c8147429d8406cc63 --hard
wenet_path=$(pwd)
```

路径说明：

${wenet_path}表示wenet开源模型代码的路径

${code_path}表示modelzoo中Wenet_for_Pytorch工程代码的路径，例如code_path=/home/ModelZoo-PyTorch/ACL_PyTorch/built-in/audio/Wenet_for_Pytorch

4. 下载网络权重文件并导出onnx

下载链接：http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210601_u2pp_conformer_exp.tar.gz

下载压缩文件，将文件解压，将文件夹内的文件放置到wenet/examples/aishell/s0/exp/conformer_u2文件夹下，若没有该文件夹，则创建该文件夹

```
tar -zvxf 20210601_u2pp_conformer_exp.tar.gz
mkdir -p ${wenet_path}/examples/aishell/s0/exp/conformer_u2
cp -r 20210601_u2pp_conformer_exp/20210601_u2++_conformer_exp/* ${wenet_path}/examples/aishell/s0/exp/conformer_u2
```

5. 拷贝本工程下提供的diff、sh、py文件到wenet对应目录下

   ```
   cd  ${code_path}
   cp -r *diff ${wenet_path}
   cp -r *.sh ${wenet_path}/examples/aishell/s0
   cp -r export_onnx.py ${wenet_path}/wenet/bin/
   cp -r process_encoder_data_noflash.py ${wenet_path}/wenet/bin/
   cp -r process_encoder_data_flash.py ${wenet_path}/wenet/bin/
   cp -r recognize_attenstion_rescoring.py ${wenet_path}/wenet/bin/
   cp -r static.py ${wenet_path}/wenet/bin/
   cp -r slice_helper.py ${wenet_path}/wenet/transformer
   cp -r acl_net.py ${wenet_path}/wenet/transformer
   ```

6. 数据集下载

   ```
   cd ${wenet_path}/examples/aishell/s0/
   bash run.sh --stage -1 --stop_stage -1 # 下载数据集
   bash run.sh --stage 0 --stop_stage 0 # 处理数据集
   bash run.sh --stage 1 --stop_stage 1 # 处理数据集
   bash run.sh --stage 2 --stop_stage 2 # 处理数据集
   bash run.sh --stage 3 --stop_stage 3 # 处理数据集
   ```



## 2 模型转换

1. 导出onnx

```
cd ${wenet_path}
patch -p1 < export_onnx.diff
bash export_onnx.sh exp/conformer_u2/train.yaml exp/conformer_u2/final.pt
```

运行导出onnx文件并保存${wenet_path}/examples/aishell/s0/onnx/文件夹下

2.  运行脚本将onnx转为om模型

   om_gener工具修改onnx模型，生成decoder_final.onnx、encoder_revise.onnx、no_flash_encoder_revise.onnx，并运行相应脚本生成om模型，注意配置环境变量

   ${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```
cd ${code_path}
cp ${wenet_path}/examples/aishell/s0/onnx/* ${code_path}/
python3 adaptdecoder.py
python3 adaptencoder.py
python3 adaptnoflashencoder.py
bash encoder.sh Ascend${chip_name} 
bash decoder.sh Ascend${chip_name}
bash no_flash_encoder.sh Ascend${chip_name} # Ascend310P3
```

## 3 离线推理 

### 	动态shape场景：

​        1. 设置日志等级 export ASCEND_GLOBAL_LOG_LEVEL=3

​        2. 拷贝om模型到${wenet_path}/examples/aishell/s0

```
cp ${code_path}/decoder_final.om ${wenet_path}/examples/aishell/s0/
cp ${code_path}/encoder_revise.om ${wenet_path}/examples/aishell/s0/
cp ${code_path}/no_flash_encoder_revise.om ${wenet_path}/examples/aishell/s0/
```

#### 非流式场景

- 非流式场景下encoder处理数据


```
cd ${wenet_path}
git checkout .
patch -p1 < get_no_flash_encoder_out.diff
cd ${wenet_path}/examples/aishell/s0/
bash run_no_flash_encoder_out.sh --bin_path encoder_data_noflash --model_path ./no_flash_encoder_revise.om --json_path encoder_noflash.json --perf_json t1.json
```

- 获取非流式场景下decoder处理结果：



```
cd ${wenet_path}
git checkout .
patch -p1 < getwer.diff
cd ${wenet_path}/examples/aishell/s0/
bash run_attention_rescoring.sh --model_path ./decoder_final.om --perf_json t2.json --json_path encoder_noflash.json --bin_path encoder_data_noflash
```

- 查看overall精度

  ```
  cat ${wenet_path}/examples/aishell/s0/exp/conformer/test_attention_rescoring/wer | grep "Overall"
  ```

- 查看非流式性能

  t1.json为encoder耗时，t2.json为decoder耗时，非流式性能为encoder耗时和decoder耗时的总和

  ```
  cp ${wenet_path}/examples/aishell/s0/t1.json ${code_path}
  cp ${wenet_path}/examples/aishell/s0/t2.json ${code_path}
  cd ${code_path}
  python3.7.5 infer_perf.py
  ```

  

#### 流式场景

- 获取流式场景下encoder处理数据

```
cd ${wenet_path}
git checkout .
patch -p1 < get_flash_encoder_out.diff
cd ${wenet_path}/examples/aishell/s0/
bash run_encoder_out.sh --bin_path encoder_data --model_path encoder_revise.om --json_path encoder.json
```

- 获取流式场景下，decoder处理结果：


首先cd到wenet根目录下

```
cd ${wenet_path}
git checkout .
patch -p1 < getwer.diff
cd ${wenet_path}/examples/aishell/s0/
bash run_attention_rescoring.sh --model_path ./decoder_final.om --json_path  encoder.json --bin_path ./encoder_data
```

- 查看overall精度

  ```
  cat ${wenet_path}/examples/aishell/s0/exp/conformer/test_attention_rescoring/wer | grep "Overall"
  ```

### **评测结果：**   

| 模型  |          官网pth精度           |     310P/310离线推理精度     | gpu性能 | 310P性能 | 310性能 |
| :---: | :----------------------------: | :-------------------------: | :-----: | :-----: | ------- |
| wenet | GPU流式：5.94%， 非流式：4.64% | 流式：5.66%， 非流式：4.78% |         |  7.69   | 11.6fps |



### 静态shape场景(仅支持非流式场景)：

- onnx转om:


```
cd ${code_path}
bash static_encoder.sh
bash static_decoder.sh
cp ${code_path}/encoder_fendang_262_1478_static.om ${wenet_path}/
cp ${code_path}/decoder_fendang.om ${wenet_path}/
```

- 精度测试:


```
cd ${wenet_path}/
git checkout .
patch -p1 < acc.diff
cd ${wenet_path}/examples/aishell/s0/
bash run_static.sh
```

- 查看overall精度

  ```
  cat ${wenet_path}/examples/aishell/s0/exp/conformer/test_attention_rescoring/wer | grep "Overall"
  ```

- 查看性能

  性能为encoder + decoder 在数据集上的平均时间

```
cat ${wenet_path}/examples/aishell/s0/exp/conformer/test_attention_rescoring/text | grep "FPS"
```

