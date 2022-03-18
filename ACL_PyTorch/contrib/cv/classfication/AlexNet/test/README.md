环境准备：

1.数据集路径
本模型数据集放在/opt/npu/下

2.进入工作目录
cd AlexNet

3.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时 **不推荐**使用该命令安装
pip3.7 install -r requirements.txt

4.获取模型代码
git clone https://github.com/pytorch/vision

5.获取权重文件
wget https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth

6.获取benchmark工具

将 benchmark工具放在当前目录下

7.310上执行，执行时确保device空闲

bash test/pth2om.sh

bash test/eval_acc_perf.sh --datasets_path=/opt/npu/
