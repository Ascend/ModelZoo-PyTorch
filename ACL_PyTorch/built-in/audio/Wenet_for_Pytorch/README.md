# Wenet模型PyTorch离线推理指导

## 1 环境准备 

1. 安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3.7 install -r requirements.txt  
```

其他需要安装的请自行安装

1. 获取，修改与安装开源模型代码  

```
git clone https://github.com/wenet-e2e/wenet.git
cd wenet
git reset 9c4e305bcc24a06932f6a65c8147429d8406cc63 --hard
```

3. 下载网络权重文件并导出onnx

下载链接：http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210601_u2pp_conformer_exp.tar.gz

下载压缩文件，将文件解压，将文件夹内的文件放置到wenet/examples/aishell/s0/exp/conformer_u2文件夹下，若没有该文件夹，则创建该文件夹

首先将所有提供的diff文件放到wenet根目录下，patch -p1 < export_onnx.diff文件适配导出onnx的代码，将提供的export_onnx.py、process_encoder_data_flash.py、process_encoder_data_noflash.py、recognize_attenstion_rescoring.py、static.py文件放到wenet/wenet/bin/目录下，将提供的slice_helper.py, acl_net.py文件放到wenet/wenet/transformer文件夹下，将提供的sh脚本文件放到wenet/examples/aishell/s0/目录下，运行bash export_onnx.sh exp/conformer_u2/train.yaml exp/conformer_u2/final.pt导出onnx文件在当前目录下的onnx文件夹下

4.  运行脚本将onnx转为om模型

首先使用改图工具om_gener改图，该工具链接为https://gitee.com/liurf_hw/om_gener，

安装之后，将生成的onnx放至修改脚本文件同一目录，使用以下命令修改脚本，

python3 adaptdecoder.py生成decoder_final.onnx

python3 adaptencoder.py生成encoder_revise.onnx

python3 adaptnoflashencoder.py生成no_flash_encoder_revise.onnx

配置环境变量，使用atc工具将模型转换为om文件，命令参考提供的encoder.sh, decoder.sh, no_flash_encoder.sh脚本，运行即可生成对应的om文件，若设备为710设备，修改sh脚本中的

--soc_version=Ascend710即可

5. 数据集下载：

   在wenet/examples/aishell/s0/文件夹下运行

   bash run.sh --stage -1 –stop_stage -1下载数据集

   运行bash run.sh --stage 0 --stop_stage 0处理数据集

   运行bash run.sh --stage 1 --stop_stage 1处理数据集

   运行bash run.sh --stage 2 --stop_stage 2处理数据集
   
   运行bash run.sh --stage 3 --stop_stage 3处理数据集

## 2 离线推理 

​	动态shape场景：

   首先export ASCEND_GLOBAL_LOG_LEVEL=3

1. (1)非流式场景精度获取

   获取非流式场景下encoder处理数据：cd到wenet根目录下
   
   ```
   git checkout .
   patch -p1 < get_no_flash_encoder_out.diff
   cd examples/aishell/s0/
   bash run_no_flash_encoder_out.sh
   ```
   
   以上步骤注意，wenet/bin/process_encoder_data_noflash.py文件中--bin_path， --model_path，--json_path分别保存encoder生成的bin文件，非流式encoder om模型位置，encoder生成bin文件的shape信息。
   
   获取非流式场景下，decoder处理结果：cd到wenet根目录下
   
   ```
   git checkout .
   patch -p1 < getwer.diff
   cd examples/aishell/s0/
   bash run_attention_rescoring.sh
   ```
   
   注意wenet/bin/recognize_attenstion_rescoring.py文件中--bin_path， --model_path， --json_path分别是非流式encoder om生成bin文件，即上一步生成的bin文件路径，decoder模型om路径，非流式encoder生成bin文件shape信息对应的json文件，即上一步生成的json文件。查看wenet/examples/aishell/s0/exp/conformer/test_attention_rescoring/wer文件的最后几行，即可获取overall精度
   
    (2) 流式场景精度获取
   
   ​	获取非流式场景下encoder处理数据：cd到wenet根目录下
   
   ```
   git checkout .
   patch -p1 < get_flash_encoder_out.diff
   cd examples/aishell/s0/
   bash run_encoder_out.sh
   ```
   
   以上步骤注意，wenet/bin/process_encoder_data_flash.py文件中--bin_path, --model_path, --json_path分别保存encoder生成的bin文件， 模型路径信息，encoder生成bin文件的shape信息；
   
   
   
   获取流式场景下，decoder处理结果：cd到wenet根目录下
   
   ```
   git checkout .
   patch -p1 < getwer.diff
   cd examples/aishell/s0/
   bash run_attention_rescoring.sh
   ```
   
   注意wenet/bin/recognize_attenstion_rescoring.py文件中--bin_path， --model_path， --json_path分别是非流式encoder om生成bin文件，即上一步生成的bin文件路径，decoder模型om路径，流式encoder生成bin文件shape信息对应的json文件，即上一步生成的json文件。查看wenet/examples/aishell/s0/exp/conformer/test_attention_rescoring/wer文件的最后，即可获取overall精度。流式场景下测试速度较慢，可以在encoder.py文件中的BaseEncoder中修改，chunk_xs = xs[:, cur:end, :]修改为chunk_xs = xs[:, cur: num_frames, :]，同时在for循环最后offset += y.size(1)后面一行加上break**评测结果：**   

| 模型  |          官网pth精度           |     710/310离线推理精度     | gpu性能 | 710性能 | 310性能 |
| :---: | :----------------------------: | :-------------------------: | :-----: | :-----: | ------- |
| wenet | GPU流式：5.94%， 非流式：4.64% | 流式：5.66%， 非流式：5.66% |         |  7.69   | 11.6fps |

生成的t1.json, t2.json文件中分别为encoder，decoder耗时，将其相加即可，运行python3.7.5 infer_perf.py

静态shape场景(仅支持非流式场景)：

onnx转om:

```
bash static_encoder.sh
bash static_decoder.sh
```

精度测试:

首先export ASCEND_GLOBAL_LOG_LEVEL=3，指定acc.diff中self.encoder_ascend， self.decoder_ascend加载的文件为静态转出的encoder，decoder模型，修改run.sh中average_checkpoint为false, decode_modes修改为attention_rescoring， stage=5 decode阶段185、198行修改python为python3.7.5, 185行recognize.py修改为static.py

```
git checkout .
patch -p1 < acc.diff
cd examples/aishell/s0/
bash run.sh --stage 5 --stop_stage 5
```

性能：在wenet/examples/aishell/s0/exp/conformer/test_attention_rescoring/text文件最后一行有FPS性能数据

