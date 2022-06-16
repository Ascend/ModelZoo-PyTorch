# ECAPA-TDNN模型Pytorch离线推理指导


## 1.环境准备

### 1.1 深度学习框架

```
CANN == 5.0.2
torch == 1.7.1
torchvision == 0.8.2
onnx 
```
### 1.2 python第三方库

```
tqdm
scipy
librosa
matplotlib
pydub
tensorboardX
onnx
onnxruntime
skl2onnx
```

```
conda create -n tdnn python==3.7
pip install -r requirements.txt
```



### 1.3 onnx优化工具onnx_tool

```
git clone https://gitee.com/zheng-wengang1/onnx_tools.git
cd onnx_tools && git checkout cbb099e5f2cef3d76c7630bffe0ee8250b03d921
cd ..
```

### 1.4 获取开源模型代码

```
git clone --recursive https://github.com/Joovvhan/ECAPA-TDNN.git
mv ecapa-tdnn ECAPA_TDNN
```

### 1.5 准备数据集
用户需自行获取VoxCeleb1数据集中测试集（无需训练集），上传数据集到服务器中,必须要与preprocess.py同目录.
其内容如下所示

![输入图片说明](image3.png)



### 1.6 [获取msame工具](https://gitee.com/ascend/tools/tree/master/msame)
将msame文件放到当前工作目录

```
git clone https://gitee.com/ascend/tools.git
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub
cd tools/msame
./build.sh g++ out
cd out
mv main ../../../msame
```




## 2.模型转换

### 2.1 pytorch模型转onnx模型
加载当前工作目录下权重文件即Ecapa_Tdnn/checkpoint/,该权重为自己训练出的权重，后续精度以该权重下精度为标准

获取基准精度，作为精度对比参考， checkpoint为权重文件相对路径， VoxCeleb为数据集相对路径， batch_size = 4

```
python get_originroc.py checkpoint VoxCeleb 4
```



利用权重文件和模型的网络结构转换出所需的onnx模型， checkpoint为权重文件相对路径， ecapa_tdnn.onnx 为生成的onnx模型相对路径

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$PYTHONPATH:./ECAPA_TDNN
export PYTHONPATH=$PYTHONPATH:./ECAPA_TDNN/tacotron2
python pytorch2onnx.py checkpoint ecapa_tdnn.onnx 
```

将转化出的onnx模型进行优化， ecapa_tdnn.onnx为优化前onnx模型， ecapa_tdnn_sim.onnx为优化后onnx模型

```
python fix_conv1d.py ecapa_tdnn.onnx ecapa_tdnn_sim.onnx
```

### 2.2 onnx模型转om模型，以batch_size=16为例
在310P环境下，运行to_om.sh脚本，其中--model和--output参数自行修改，下面仅作参考

${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```
sudo apt install dos2unix
dos2unix ./*.sh
chmod +x *.sh
./to_om.sh Ascend${chip_name}
```

## 3.数据集预处理

在当前工作目录下，执行以下命令行,其中VoxCeleb为数据集相对路径，input/为模型所需的输入数据相对路径，speaker/为后续后处理所需标签文件的相对路径,batch_size = 4

```
python preprocess.py VoxCeleb input_bs4/ speaker_bs4/ 4
```

执行完成后将Ecapa_Tdnn/input_bs4/下内容310P环境中

## 4.模型推理

在310P环境中，cd至msame文件夹下含.masame文件的路径下

执行推理，其中--model为之前转化好的bs为4的om模型，--input为第三步中得到的前处理后的数据路径

```
./msame --model "om/ecapa_tdnn_bs4.om" --input "input_bs4/" --output "result" --outfmt TXT
```

在生成的结果只需获取其中后缀为_0的文件，运行一下命令将其放入一个文件夹(命名为output_bs4)，其中第一个参数为batchsize

```
./generate_output.sh 4
```


## 5.生成推理精度

根据第四步中获取的结果output_bs4/和第三步中产生的speaker_bs4/标签文件，得到推理精度

```
 python postprocess.py output_bs4/ speaker_bs4/ 4 4648
```

## 6.精度对比
以下为测试出的batch_size=1和4的精度对比：

```
roc_auc:
          om          pth
bs1	  0.9866      0.9914
bs4	  0.9866      0.9908
```
精度下降不超过百分之一，精度达标

## 7.性能对比

### 7.1 GPU性能数据
**注意：**

> 测试gpu性能要确保device空闲，使用nvidia-smi命令可查看device是否在运行其它推理任务

以bs=4为例，这里的infer_cpu.onnx为优化前onnx模型

```
trtexec --onnx=ecapa_tdnn.onnx --fp16 --shapes=mel:4x80x200
```


### 7.2 NPU性能数据

以bs=4为例。利用msame进行纯推理

```
 ./msame --model "om/ecapa_tdnn_bs4.om" --output "result" --outfmt TXT --loop 100
```

| Model      | batch_size | T4Throughput/Card | 310PThroughput/Card |
|------------|------------|-------------------|--------------------|
| ECAPA-TDNN | 1          | 485.43            | 764.52             |
| ECAPA-TDNN | 4          | 705.46            | 1408.45            |
| ECAPA-TDNN | 8          | 798.4             | 1408.43            |
| ECAPA-TDNN | 16         | 770.89            | 1315.78            |
| ECAPA-TDNN | 32         | 828.84            | 1281.53            |
| ECAPA-TDNN | 64         | 847.37            | 1221.6             |
| ECAPA-TDNN | best       | 847.37            | 1408.45            |

1408.46/847.37 = 1.66 ,性能达标

