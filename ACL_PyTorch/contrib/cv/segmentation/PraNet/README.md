# PraNet模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```


2.获取，安装开源模型代码。

```shell
git clone https://github.com/DengPingFan/PraNet.git -b master
cd PraNet
git reset --hard f697d5f566a4479f2728ab138401b7476f2f65b9
patch -p1 < ../PraNet_perf.diff
因开源代码仓使用matlab评测，故需从(https://github.com/plemeri/UACANet)获取pytorch实现的评测脚本eval_functions.py，并将其放在utils目录下
将./lib/PraNet_Res2Net.py的res2net50_v1b_26w_4s(pretrained=True)修改为res2net50_v1b_26w_4s(pretrained=False)
cd ..
```
3.获取权重文件  
[PraNet训练的pth权重文件](https://drive.google.com/file/d/1pUE99SUQHTLxS9rabLGe_XTDwfS6wXEw/view)

4.数据集     
[kvasir](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view)获取TestDataset.zip并解压出Kvasir，将其放到/root/datasets/目录下，即/root/datasets/Kvasir

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  
将benchmark.x86_64或benchmark.aarch64放到当前目录  

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  
```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets/Kvasir
```
 **评测结果：**   
| 模型      | 在线推理精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  |
| PraNet bs1 | mDec:0.894;mIoU:0.836 | mDec:0.894;mIoU:0.836 |  265.0909fps | 199.5684fps |
| PraNet bs16 | mDec:0.894;mIoU:0.836 | mDec:0.894;mIoU:0.836 | 435.5139fps | 237.9004fps |



