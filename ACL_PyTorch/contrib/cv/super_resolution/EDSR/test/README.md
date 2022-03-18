环境准备：

1.数据集路径
通用的数据集统一放在/root/datasets/或/opt/npu/
本模型数据集放在/root/datasets/

2.进入工作目录
cd EDSR

3.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装
pip3.7 install -r requirements.txt

4.获取，修改与安装开源模型代码
git clone https://github.com/sanghyun-son/EDSR-PyTorch
cd EDSR-PyTorch
修改了模型代码，交付了edsr.diff
patch -p1 < ../edsr.diff

5.获取权重文件
https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt

6.获取benchmark工具
将benchmark放在当前目录

7.310上执行，执行时确保device空闲
bash test/pth2om.sh
bash test/eval_acc_perf.sh --datasets_path=/root/datasets

8.在t4环境上将onnx文件与perf_t4.sh放在同一目录
然后执行bash perf_t4.sh，执行时确保gpu空闲
