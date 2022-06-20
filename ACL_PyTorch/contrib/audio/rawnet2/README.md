### 1.环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

```
pip3 install -r requirements.txt
```

2.获取，开源模型代码

```
git clone https://github.com/Jungjee/RawNet.git
cd RawNet
patch -p1 < ../rawnet2.patch
cd ..
```

3.获取权重文件

通过2获得代码仓后，权重文件位置：RawNet\python\RawNet2\Pre-trained_model\rawnet2_best_weights.pt，将其放到当前目录

4.获取数据集 [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) ，下载Audio files测试集，重命名VoxCeleb1，确保VoxCeleb1下全部为id1xxxx文件夹，放到/root/datasets目录,注：该路径为绝对路径

5.获取 [msame工具](https://gitee.com/ascend/tools/tree/master/msame) 和 [benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)

将msame和benchmark.x86_64放到与test文件夹同级目录下。

### 2.离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲

备注：

1.需要对onnx模型进行onnxsim优化，否则无法达到精度要求，pth2om.sh脚本首先将pth文件转换为onnx模型，然后分别对bs1和bs16进行onnxsim优化，最后分别转化为om模型

2.eval_acc_perf.sh脚本逐步完成数据前处理bin文件输出、bs1和bs16模型推理、bs1和bs16精度测试，以及bs1和bs16benchmark的性能测试

```
bash test/pth2om.sh  

bash test/eval_acc_perf.sh --datasets_path=/root/datasets/VoxCeleb1/
```

评测结果：

| 模型                  | 官网pth精度                                     | 310精度   | 基准性能 | 310性能 |
| --------------------- | ----------------------------------------------- | --------- | -------- | ------- |
| Baseline-RawNet2 bs1  | [EER 2.49%](https://github.com/Jungjee/RawNet/) | EER 2.50% | 285.7fps | 72.8fps |
| Baseline-RawNet2 bs16 | [EER 2.49%](https://github.com/Jungjee/RawNet/) | EER 2.50% | 489.3fps | 77.6fps |

