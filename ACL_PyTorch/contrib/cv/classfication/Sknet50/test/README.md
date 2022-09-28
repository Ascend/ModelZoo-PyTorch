环境准备：  

1.数据集路径  
数据集统一放在/root/datasets/或/opt/npu/  
本模型数据集放在/opt/npu/

2.进入工作目录  
cd Sknet50

3.安装必要的依赖  
pip3.7 install -r requirements.txt 

4.获取模型代码  
git clone https://github.com/implus/PytorchInsight

5.如果使用补丁文件修改了模型代码则将补丁打入模型代码,如果需要引用模型代码仓的类或函数通过sys.path.append()添加搜索路径。

5.获取权重文件  
[SK-ResNet50预训练pth权重文件(百度网盘，提取码：tfwn)](https://pan.baidu.com/s/1Lx5CNUeRQXOSWjzTlcO2HQ)

7.获取benchmark工具  
将benchmark.x86_64放在当前目录  

8.310上执行，执行时确保device空闲  
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/opt/npu  
