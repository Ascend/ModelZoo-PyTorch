# Wenet模型PyTorch离线推理指导

## 1 环境准备 

1. 安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3.7 install -r requirements.txt  
```

2. ctc_dcoder:

   https://github.com/Slyne/ctc_decoder
   
   ```
   git clone https://github.com/Slyne/ctc_decoder.git
   apt-get update
   apt-get install swig
   apt-get install python3-dev 
   cd ctc_decoder/swig && bash setup.sh
   ```

其他需要安装的请自行安装

3. 获取开源模型代码  

```
git clone https://github.com/wenet-e2e/wenet.git
cd wenet
git checkout v2.0.1
wenet_path=$(pwd)
```

路径说明：

${wenet_path}表示wenet开源模型代码的路径

${code_path}表示modelzoo中Wenet_for_Pytorch工程代码的路径，例如code_path=/home/ModelZoo-PyTorch/ACL_PyTorch/built-in/audio/Wenet_for_Pytorch

4. 数据集预处理

   cd ${wenet_path}/examples/aishell/s0/
   bash run.sh --stage -1 --stop_stage -1 # 下载数据集
   bash run.sh --stage 0 --stop_stage 0 # 处理数据集
   bash run.sh --stage 1 --stop_stage 1 # 处理数据集
   bash run.sh --stage 2 --stop_stage 2 # 处理数据集
   bash run.sh --stage 3 --stop_stage 3 # 处理数据集

5. 下载网络权重文件并导出onnx

下载链接：https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.md

选择aishell数据集对应的Checkpoint Model下载即可

下载压缩文件，将文件解压，将文件夹内的文件放置到wenet/examples/aishell/s0/exp/20210601_u2++_conformer_exp文件夹下，若没有该文件夹，则创建该文件夹

```
tar -zvxf 20210601_u2++_conformer_exp.tar.gz
mkdir -p ${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp
mkdir -p ${wenet_path}/exp/20210601_u2++_conformer_exp
mkdir -p ${wenet_path}/examples/aishell/s0/onnx/
cp -r 20210601_u2pp_conformer_exp/20210601_u2++_conformer_exp/* ${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp
cp -r examples/aishell/s0/data/test/data.list ${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/
cp -r examples/aishell/s0/data/test/text ${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/
cp -r ${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/global_cmvn ${wenet_path}/exp/20210601_u2++_conformer_exp
```

6. [获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)

```
cp -r benchmark ${wenet_path}/examples/aishell/s0/ 
```

7. 拷贝本工程下提供的py文件到wenet对应目录下

```
cd  ${code_path}
cp -r export_onnx_npu.py ${wenet_path}/wenet/bin/
cp -r recognize_om.py ${wenet_path}/wenet/bin/
cp -r cosine_similarity.py ${wenet_path}/examples/aishell/s0/ 
cp -r adaptdecoder.py ${wenet_path}/examples/aishell/s0/
cp -r *.sh ${wenet_path}/examples/aishell/s0/ 
```




## 2 模型转换

1. 导出onnx

```
#非流式
python3 wenet/bin/export_onnx_npu.py --config ${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/train.yaml --checkpoint ${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/final.pt --output_onnx_dir ./onnx/ --num_decoding_left_chunks 4 --reverse_weight 0.3 
#流式
python3 wenet/bin/export_onnx_npu.py --config ${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/train.yaml --checkpoint ${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/final.pt --output_onnx_dir ./onnx/ --num_decoding_left_chunks 4 --reverse_weight 0.3 --streaming
```

运行导出onnx文件并保存${wenet_path}/examples/aishell/s0/onnx/文件夹下

2.  运行脚本将onnx转为om模型

   运行相应脚本生成om模型，注意配置环境变量

   ${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```
cd ${code_path}
cp ${wenet_path}/examples/aishell/s0/onnx/* ${code_path}/
bash online_encoder.sh Ascend${chip_name} 
bash offline_encoder.sh Ascend${chip_name} # Ascend310P3
bash static_encoder.sh Ascend${chip_name}
#若需要decoder部分,对于分档场景
python3 adaptdecoder.py
bash static_decoder.sh Ascend${chip_name}
```

## 3 离线推理 

run.sh脚本封装了ctc_greedy_search场景的动态shape和动态分档的推理，并将性能和精度分别保存在了offline_test_result.txt online_test_result.txt以及static_test_result.txt中，

运行bash run.sh Ascend310P3即可

### 	动态shape场景：

​        1. 设置日志等级 export ASCEND_GLOBAL_LOG_LEVEL=3

​        2. 拷贝om模型到${wenet_path}/examples/aishell/s0

```
cp ${code_path}/online_encoder.om ${wenet_path}/examples/aishell/s0/
cp ${code_path}/offline_encoder.om ${wenet_path}/examples/aishell/s0/
```

#### 非流式场景

- 非流式场景下推理模型


```
python3 wenet/bin/recognize_om.py --config=${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/train.yaml --test_data=${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/data.list --dict=units.txt --mode=ctc_greedy_search --result_file=offline_res_result.txt --encoder_om=encoder_offline.om --decoder_om=xx.om --batch_size=1 --device_id=0 --test_file=offline_test_result.txt
```

- 计算并查看overall精度

  ```
  python3 tools/compute-wer.py --char=1 --v=1 text offline_res_result.txt > offline_wer
  cat offline_wer | grep "Overall"
  ```

- 查看非流式性能

  性能和精度保存在offline_dynamic_results.txt


#### 流式场景

- 获取流式场景下性能

  直接用benchmark测试，例如对于x86场景

  ./benchmark.x86_x64 -batch_size=64 -om_path=encoder_online.om -round=1000 -device_id=0

- 查看精度(余弦相似度)

  ```
  python3 cosine_similarity.py
  ```

### 动态分档场景(非流式场景)：

- 推理模型:


```
python3 wenet/bin/recognize_om.py --config=${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/train.yaml --test_data=${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/data.list --dict=units.txt --mode=ctc_greedy_search --result_file=static_res_result.txt --encoder_om=encoder_static.om --decoder_om=decoder_static.om --batch_size=32--device_id=0 --static --test_file=static_test_result.txt
```

- 查看性能和精度：

  在static_test_result.txt文件中

以上步骤可直接运行run.sh脚本



