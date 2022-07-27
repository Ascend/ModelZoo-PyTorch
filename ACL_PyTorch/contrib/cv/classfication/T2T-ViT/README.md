# T2T-ViT Onnx模型端到端推理指导

## 1 模型概述


### 1.1 论文地址

[Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](https://arxiv.org/abs/2101.11986)

### 1.2 代码地址

开源仓：[https://github.com/yitu-opensource/T2T-ViT](https://github.com/yitu-opensource/T2T-ViT)<br>
branch：main<br>
commit_id：0f63dc9558f4d192de926504dbddfa1b3f5db6ca<br>

## 2 环境说明

该模型离线推理使用 Atlas 300I Pro 推理卡，所有步骤都在 [CANN 5.1.RC1](https://www.hiascend.com/software/cann/commercial) 环境下进行，CANN包以及相关驱动、固件的安装请参考 [软件安装](https://www.hiascend.com/document/detail/zh/canncommercial/51RC1/envdeployment/instg)。
### 2.1 安装依赖
```shell
conda create -n ${env_name} python=3.7.5
conda activate ${env_name}
pip install -r requirements.txt 
```

### 2.2 获取开源仓代码
```shell
git clone https://github.com/yitu-opensource/T2T-ViT.git
cd T2T-ViT
git checkout main
git reset --hard 0f63dc9558f4d192de926504dbddfa1b3f5db6ca
```

## 3 源码改动
1.pytorch在1.8版本之后container_abcs就已经被移除，因此在使用timm时会出现错误

因此需要修改timm包内models/layers文件夹中的helpers.py文件第6行：
```
import collections.abc as container_abcs
```
2.由于310P上无GPU，因此在使用timm时会出现错误
因此需要修改timm包内data文件夹中的loader.py文件第66、67行和第78行的__iter__：
```
self.mean = torch.tensor([x * 255 for x in mean]).view(1, 3, 1, 1) 
self.std = torch.tensor([x * 255 for x in std]).view(1, 3, 1, 1)

def __iter__(self):
        first = True
        for next_input, next_target in self.loader:
            if self.fp16:
                next_input = next_input.half().sub_(self.mean).div_(self.std)
            else:
                next_input = next_input.float().sub_(self.mean).div_(self.std)
            if self.random_erasing is not None:
                next_input = self.random_erasing(next_input)

            if not first:
                yield input, target
            else:
                first = False
            input = next_input
            target = next_target

        yield input, target 
```
3.由于onnx模型中的add算子输入的一个常量数值非常小，在float32情况下可以正常表示，但float16无法正常表示这个常量，导致推理出现精度问题
因此需要修改models文件夹内token_performer.py文件：
添加一个函数forgr_einsum：
```
def forge_einsum(equation, a, b):
    if equation == 'bti,bi->bt':
        return torch.sum(a * b.unsqueeze(1), dim=2)
    elif equation == 'bti,bni->btn':
        return torch.sum(a.unsqueeze(2) * b.unsqueeze(1), dim=3)
    elif equation == 'bti,mi->btm':
        return torch.sum(a.unsqueeze(2) * b.unsqueeze(0), dim=3)
    elif equation == 'bin,bim->bnm':
        return torch.sum(a.unsqueeze(3) * b.unsqueeze(2), dim=1)
    else:
        raise Exception('Unkown equation')
```
并修改第41、48、49、50行：
```
wtx = forge_einsum('bti,mi->btm', x.float(), self.w)

D = forge_einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)
kptv = forge_einsum('bin,bim->bnm', v.float(), kp)
y = forge_einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)
```


## 4 模型转换

### 4.1 Pytorch转ONNX模型

1. 下载pth权重文件
```shell
wget https://github.com/yitu-opensource/T2T-ViT/releases/download/main/81.5_T2T_ViT_14.pth.tar
```

2. 执行T2T_ViT_pth2onnx.py脚本，生成ONNX模型文件

```shell
python3.7 T2T_ViT_pth2onnx.py --pth-dir ${pth_dir} --onnx-dir ${onnx_dir}
```
参数说明：<br>
--pth-dir: Pytorch模型文件路径<br>
--onnx-dir: ONNX模型文件保存路径（包括文件名）<br>

### 4.2 ONNX转OM模型

1. 设置环境变量

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

该命令中使用CANN默认安装路径(/usr/local/Ascend/ascend-toolkit)中的环境变量，使用过程中请按照实际安装路径设置环境变量。

2、生成OM模型
ATC工具的使用请参考 [ATC模型转换](https://www.hiascend.com/document/detail/zh/canncommercial/51RC1/inferapplicationdev/atctool)

```shell
atc --framework=5 --model=${onnx-path} --output=${om-path} --input_format=NCHW --input_shape="image:${bs},3,224,224" --log=error --soc_version=Ascend${chip_name} --keep_dtype=keep_dtype.cfg
```
参数说明：<br>
--model：ONNX模型文件路径<br>
--output 生成OM模型的保存路径（含文件名）<br>
执行命令前，需设置--input_shape参数中bs的数值，例如：1、4、8、16、32、64 <br> 
chip_name可通过`npu-smi info`指令查看，例：310P3<br>
![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

## 5 数据预处理


### 5.1 数据集获取

该模型使用[ImageNet官网](http://www.image-net.org/)的5万张验证集进行测试
数据集结构如下：
```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### 5.2 数据集预处理

1.备份一份开源仓的main.py文件，并命名为main_copy.py，用于生成前后处理文件。<br>
2. 生成预处理脚本T2T_ViT_preprocess.py
```shell
patch -p1 main.py T2T_ViT_preprocess.patch
```
得到的main.py重命名为T2T_ViT_preprocess.py。<br>
3. 执行预处理脚本，生成数据集预处理后的bin文件

```shell
python3.7 T2T_ViT_preprocess.py -–data-dir ${dataset_path} --out-dir ${prep_output_dir} –gt-path ${groundtruth_path} -–batch-size ${batchsize}
```
参数说明：<br>
--data-dir：数据集路径<br>
--out-dir：保存bin文件路径<br>
--gt-path：保存标签文件路径<br>
--batch-size：需要测试的batchsize<br>


## 6 离线推理

### 6.1 msame工具

本项目使用msame工具进行推理，msame编译及用法参考[msame推理工具](https://gitee.com/ascend/tools/tree/master/msame)。<br>
msame推理前需要设置环境变量：
``` shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
```

该命令中使用CANN默认安装路径(/usr/local/Ascend/ascend-toolkit)中的环境变量，使用过程中请按照实际安装路径设置环境变量。

### 6.2 离线推理

运行如下命令进行离线推理：

```shell
./msame --model ${om_path}  --input ${input_dataset_path} --output ${output_dir} --outfmt BIN
```
参数说明：<br>
--model：为om模型文件路径<br>
--input：数据预处理后的bin文件路径<br>
--output：保存标签路径（含文件名）<br>

输出结果保存在output_dir中，文件类型为bin文件。

### 6.3 精度验证

1.生成后处理脚本T2T_ViT_postprocess.py脚本
```shell
patch -p1 main_copy.py T2T_ViT_postprocess.patch
```
得到的main_copy.py重命名为T2T_ViT_postprocess.py。

2.运行T2T_ViT_postprocess.py脚本并与npy文件比对，可以获得Accuracy Top1数据

```shell
python3.7 T2T_ViT_postprocess.py –result-dir ${msame_bin_path} –gt-path ${gt_path} --batch-size ${batchsize}
```
参数说明：<br>
--result-dir：生成推理结果所在路径<br>
--gt-path：标签数据文件路径<br>
--batch-size：需要测试的batchsize<br>


### 6.4 性能验证
用msame工具进行纯推理100次，然后根据平均耗时计算出吞吐率。
```shell
./msame --model ${om_path} --output ${output_path} --outfmt TXT --loop 100
```
参数说明：<br>
--model：为om模型文件路径<br>
--output：保存推理结果路径<br>

说明：性能测试前使用`npu-smi info`命令查看 NPU 设备的状态，确认空闲后再进行测试。

执行上述脚本后，日志中会记录的除去第一次推理时间的平均时间，即为NPU计算的平均耗时(ms)。以此计算出模型在对应 batch_size 下的吞吐率：
$$ 吞吐率 = \frac {bs * 1000} {mean}$$



## 7 精度和性能对比

总结：
 1. 310P上离线推理的精度(81.414%)与Pytorch在线推理精度(81.5%)基本持平；
 2. 性能最优的batch_size为16，310P性能/性能基准=6倍。

各batchsize对比结果如下：

|     模型     |                        开源仓Pytorch精度                        | 310P离线推理精度 | 基准性能 | 310P性能 |
| :----------: | :-------------------------------------------------------: | :--------------: | :------: | :------: |
| T2T-ViT bs1  | [rank1:81.5%](https://github.com/yitu-opensource/T2T-ViT) |   rank1:81.4%    |  24fps   |  142fps  |
| T2T-ViT bs4  | [rank1:81.5%](https://github.com/yitu-opensource/T2T-ViT) |   rank1:81.4%    |  32fps   |  179fps  |
| T2T-ViT bs8  | [rank1:81.5%](https://github.com/yitu-opensource/T2T-ViT) |   rank1:81.4%    |  39fps   |  212fps  |
| T2T-ViT bs16 | [rank1:81.5%](https://github.com/yitu-opensource/T2T-ViT) |   rank1:81.4%    |  35fps   |  210fps  |
| T2T-ViT bs32 | [rank1:81.5%](https://github.com/yitu-opensource/T2T-ViT) |   rank1:81.4%    |  34fps   |  203fps  |
| T2T-ViT bs64 | [rank1:81.5%](https://github.com/yitu-opensource/T2T-ViT) |   rank1:81.4%    |  36fps   |  198fps  |