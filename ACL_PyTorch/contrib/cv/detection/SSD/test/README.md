环境准备：

1.数据集路径
通用的数据集统一放在/root/datasets/或/opt/npu/
本模型数据集放在/root/datasets/

2.进入工作目录
cd SSD

3.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装
pip3.7 install -r requirements.txt

4.获取模型代码及权重文件
bash test/prepare_env.sh

5.获取benchmark工具
将benchmark.x86_64 benchmark.aarch64放在当前目录

6.310上执行，执行时确保device空闲
bash test/pth2om.sh
bash test/eval_acc_perf.sh --datasets_path=/root/datasets

7.在基准环境上在线推理（onnx包含自定义算子，因此不能使用开源TensorRT测试性能数据，故在基准机器上使用pth在线推理测试性能数据）
bash test/perf_benchmark.sh