## RDN模型PyTorch离线推理指导

### 1 环境准备

1. 安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

   ```python
   pip3.7 install -r requirements.txt
   ```

2. 获取开源模型代码

   ```
   git clone https://github.com/yjn870/RDN-pytorch -b master
   ```

   开源模型代码仓没有安装脚本，可以通过sys.path.append(r"./RDN-pytorch")添加搜索路径，然后在pth2onnx脚本中就可以引用模型代码的函数或类

3. 获取权重文件

   [RDN_x2预训练pth权重文件](https://www.dropbox.com/s/pd52pkmaik1ri0h/rdn_x2.pth?dl=0)

4. 数据集

   开源代码仓只提供了h5格式的Set5数据集，由于代码仓评测精度的脚本采用png格式的图片作为输入，可通过[Set5](https://github.com/hengchuan/RDN-TensorFlow/tree/master/Test/Set5)下载png格式的Set5数据集，并将文件夹重命名为set5，数据集放在/root/datasets目录

5. 获取benchmark工具

   将benchmark.x86_64放到当前目录

   

### 2 离线推理

310P上执行，执行时使npu-smi info查看设备状态，确保device空闲

```
bash test/pth2om.sh 
bash test/eval_acc_perf.sh --datasets_path=/root/datasets
```



**评测结果：**

|  模型   | 官网pth精度 | 310离线推理精度 | 310P离线推理精度 |  基准性能  |  310性能   |  310P性能  |
| :-----: | :---------: | :-------------: | :--------------: | :--------: | :--------: | :--------: |
| RDN bs1 | PSNR：38.18 |   PSNR：38.27   |   PSNR：38.27    | fps:22.267 | fps:29.653 | fps:49.469 |

- 因Set5数据集只有5张图片，因此仅使用了bs1进行评测。

