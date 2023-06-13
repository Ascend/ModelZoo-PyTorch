# Wenet模型PyTorch离线推理指导

## 1 环境准备 

1. 安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3 install -r requirements.txt  
```

2. 获取，修改与安装开源模型代码  

```
git clone https://github.com/wenet-e2e/wenet.git
cd wenet
git reset 9c4e305bcc24a06932f6a65c8147429d8406cc63 --hard
```

3. 下载网络权重文件并导出onnx

下载链接：http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210601_u2pp_conformer_exp.tar.gz下载压缩文件，将文件解压，将文件夹内的文件放置到wenet/examples/aishell/s0/exp/conformer_u2文件夹下，若没有该文件夹，则创建该文件夹

将提供的diff文件放到wenet根目录下
patch -p1 < wenet_onnx.diff文件适配导出onnx的代码
将提供的export_onnx.py、static.py文件放到wenet/wenet/bin/目录下
将提供的slice_helper.py, acl_net.py文件放到wenet/wenet/transformer文件夹下
将提供的sh脚本文件（除了static_encoder.sh和static_decoder.sh）放到wenet/examples/aishell/s0/目录下
运行bash export_onnx.sh exp/conformer_u2/train.yaml exp/conformer_u2/final.pt导出onnx文件在当前目录下的onnx文件夹下

4.  修改onnx

首先使用改图工具om_gener改图，该工具链接为https://gitee.com/liurf_hw/om_gener，安装之后使用以下命令修改脚本，

python3 adaptdecoder.py生成decoder_final.onnx

python3 adaptnoflashencoder.py生成no_flash_encoder_revise.onnx

5. 数据集下载：

   在wenet/examples/aishell/s0/文件夹下运行bash run.sh --stage -1 –stop_stage -1下载数据集

   运行bash run.sh --stage 0 --stop_stage 0处理数据集

   运行bash run.sh --stage 1 --stop_stage 1处理数据集

   运行bash run.sh --stage 2 --stop_stage 2处理数据集

   运行bash run.sh --stage 3 --stop_stage 3处理数据集

## 2 离线推理 

onnx转om:
将static_encoder.sh和static_decoder.sh放到s0/onnx文件下，请以实际安装环境配置环境变量。

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash static_encoder.sh
bash static_decoder.sh
```

精度测试:

首先export ASCEND_GLOBAL_LOG_LEVEL=3，指定acc.diff中self.encoder_ascend， self.decoder_ascend加载的文件为静态转出的encoder，decoder模型路径

```
git checkout .
patch -p1 < acc.diff
cd examples/aishell/s0/
bash static.sh
```

性能：在wenet/examples/aishell/s0/exp/conformer/test_attention_rescoring/text文件最后一行有fps性能数据


gpu性能测试：

```
git checkout .
patch -p1 < gpu_fps.diff
cd examples/aishell/s0/

// 运行之前修改run.sh中average_checkpoint为false, decode_modes修改为attention_rescoring 修改stage5中ctc_weight=0.3 reverse_weight=0.3
bash run.sh --stage 5 --stop_stage 5
```
在wenet/examples/aishell/s0/exp/conformer/test_attention_rescoring/text文件最后一行有fps性能数据

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
