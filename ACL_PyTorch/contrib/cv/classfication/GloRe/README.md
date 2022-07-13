# GloRe模型PyTorch离线推理指导

## 1 环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

```
CANN 5.1.RC1
pip3.7 install -r requirements.txt  
```

2.获取，修改与安装开源模型代码

```

git clone https://github.com/facebookresearch/GloRe -b master 
cd GloRe
git reset --hard 9c6a7340ebb44a66a3bf1945094fc685fb7b730d
cd ..
```
3.[获取基于UCF101数据集训练出来的权重](https://ascend-pytorch-model-file.obs.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/GloRe/GloRe.pth)


4.[获取数据集UCF101](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)


5.[获取msame工具](https://gitee.com/ascend/tools/tree/master/msame)

6.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)

将benchmark.x86_64或benchmark.aarch64放到当前目录

## 2 模型转换

1. 获取权重文件方法。

从源码包中获取权重文件“GloRe.pth”。

2. 导出.onnx文件。

使用GloRe.pth导出onnx文件。

执行“GloRe_pth2onnx.py”脚本文件。

```
python3.7 GloRe_pth2onnx.py GloRe.pth GloRe.onnx
```

运行成功后生成“GloRe.onnx”模型文件。

​	3.使用onnxsim优化模型

```
python3.7 -m  onnxsim --input-shape "1,3,8,224,224"  GloRe.onnx GloRe.onnx
```

​	4.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

​	5.使用atc工具将onnx文件转换成om文件
${chip_name}可通过`npu-smi info`指令查看

![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```
atc --framework=5 --model=GloRe.onnx --output=GloRe_bs1 --input_format=NCHW --input_shape="image:1,3,8,224,224" --log=error --soc_version=Ascend${chip_name}
```

## 3 模型推理

​	1.获取msame工具

```
git clone https://gitee.com/ascend/tools.git
```

​	2.设置环境变量

```
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub
```

​	3.进入tools/msame文件夹，设置文件权限，运行编译脚本

```
cd tools/msame
dos2unix *.sh
chmod u+x  build.sh
./build.sh g++  /home/tools/msame/out    #路径为main所在绝对路径  
cp ./out/main   ../../   #本步骤要在“/tools/msame”文件夹目录下执行
cd ../..
```
​	4.执行推理

```
chmod u+x main
./main --model "GloRe_bs1.om" --input "b in/bs1" --output "om_res_bs1" --outfmt TXT --device 0
```

​	5.精度验证

```
python3.7 GloRe_postprocess.py --i om_res_bs1 --t bs1_target.json --o res_bs1.json 
```


| 模型      | pth精度  | 310精度  | 310P精度 | 基准性能    | 310性能    | 310P性能 |
| :------: | :------: | :------: | :------:  | :------:  | -------- | :------: |
| GloRe bs1  | top1:87.79% top5:98.02% | acc1:94.14% acc5:99.56% | acc1:94.12% acc5:99.56% |  122.4380fps | 54.48fps | 84.28fps |
| GloRe bs16 | top1:87.79% top5:98.02% | acc1:94.14% acc5:99.56% | acc1:94.12% acc5:99.56% |  148.0453fps | 57.22fps | 80.63fps |
