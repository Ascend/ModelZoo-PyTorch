# AlignedReID模型PyTorch离线推理指导

## 1. 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3.7 install -r requirements.txt  
```

2.上传源码包到服务器任意目录并解压（如：/home/AlignedReID）

```
├── benchmark.aarch64            //离线推理工具（适用ARM架构） 
├── benchmark.x86_64             //离线推理工具（适用x86架构） 
├── requirements.txt             //模型所需依赖 
├── all.patch                     //对原仓库代码做出修改的补丁
├── LICENSE            
├── perf_g.sh                     //gpu推理脚本
├── README.md  
├── Market1501_AlignedReID_300_rank1_8441.pth  //原始权重文件
├── gen_dataset_info.py         //生成数据集info文件脚本
├── AlignedReID_preprocess.py //数据集预处理脚本
├── AlignedReID_pth2onnx.py    //pth模型转onnx模型脚本（支持动态batch）
├── AlignedReID_acc_eval.py    //验证推理结果脚本，比对benchmark输出分类结果，给Accuracy 

```

2.获取，安装开源模型代码至当前目录/AlignedReID下，打上修改部分的patch

```
git clone https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch.git 
cd AlignedReID-Re-Production-Pytorch
git reset --hard 4a6ab9849c84c1186932c4c8519c689aae341f4b
patch -p1 < ../all.patch
cd ..
```

## 2.准备数据集

1.获取原始数据集

[获取Market1501](https://drive.google.com/drive/folders/1CaWH7_csm9aDyTVgjs7_3dlZIWqoBlv4)下载链接中的文件夹market1501，放在当前目录/AlignedReID下，解压/market1501中的images.tar压缩文件

```
cd market1501
tar -xvf images.tar  //tar -zxvf images.tar.gz
```

2.数据预处理

将原始数据（.jpeg）转化为二进制文件（.bin）。转化方法参考了原仓库训练预处理方法处理数据，以获得最佳精度。通过缩放、均值方差手段归一化，输出为二进制文件。

```
python3.7 AlignedReID_preprocess.py ./market1501/images ./prep_bin
```

3.生成数据集info文件

使用benchmark推理需要输入二进制数据集的info文件，用于获取数据集。输入已经得到的二进制文件，输出生成二进制数据集的info文件。

```
python3.7 gen_dataset_info.py bin ./prep_bin ./alignedreid.info 128 256
```

## 3.模型推理 

310p上执行，执行时使npu-smi info查看设备状态，确保device空闲  

1.模型转换

使用PyTorch将模型权重文件.pth转换为.onnx文件（onnx文件通用各个平台），再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

a.导出onnx文件。

```
# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 转onnx模型（支持动态batchsize）
python3.7 AlignedReID_pth2onnx.py ./Market1501_AlignedReID_300_rank1_8441.pth ./AlignedReID_bs.onnx 
```

b.使用ATC工具将.onnx文件转换为.om文件，其中batchsize依次设为1，4，8，16，32

chip_name通过npu-smi info指令查看，例如：710

![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```
atc --framework=5 --model=AlignedReID_bs.onnx --output=AlignedReID_bs${batchsize} --input_format=NCHW --input_shape="image:%{batchsize},3,256,128" --log=debug --soc_version=Ascend${chip_name} --out_nodes="Gemm_133:0;Reshape_127:0;Transpose_132:0"
```

2.开始推理验证

a.使用Benchmark工具进行推理

​	增加benchmark.{arch}可执行权限。

```
chmod u+x benchmark.x86_64
```

推理。

```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./AlignedReID_bs1.om -input_text_path=./alignedreid.info -input_width=128 -input_height=256 -output_binary=False -useDvpp=False
```

b.精度验证

调用AlignedReID_acc_eval.py脚本进行精度验证，可以获得Accuracy数据，所需验证的rank-1指标对应下图中”Computing scores for Global Distance…”后的cmc1指标，其余指标仅供参考比较。

```
python3.7 AlignedReID_acc_eval.py ./result/dumpOutput_device0
```





进行推理，验证精度和性能 **评测结果：**   

性能：

|       模型       |   310    |  310p   |     T4      | 310p-aoe |
| :--------------: | :------: | :-----: | :---------: | :------: |
| AlignedReID bs1  |  1441.7  | 1137.7  | 1016.169286 | 1391.37  |
| AlignedReID bs4  | 1930.084 | 2952.41 | 2238.839386 | 3939.99  |
| AlignedReID bs8  | 2160.46  | 3547.06 | 2517.504524 | 4723.63  |
| AlignedReID bs16 | 2150.62  | 3149.61 | 2737.996026 | 4241.01  |
| AlignedReID bs32 | 2208.228 | 2983.12 | 1529.432007 | 3717.08  |

精度：

| accuracy |  310   |  310p  |
| :------: | :----: | :----: |
|   CMC1   | 80.64% | 80.64% |
|   CMC5   | 92.10% | 92.10% |

备注：  
原仓库GPU复现精度为81%左右，310以及310p的CMC1的精度都为80.64%，相差在1%以内，精度达标，310p在bs=8的时候性能达到最高，但是310p/T4=1.2954<1.6无法达标，经过aoe操作之后310p在bs=8时达到4723.63fps，310p/T4=1.7252>1.6,固aoe之后性能达标。